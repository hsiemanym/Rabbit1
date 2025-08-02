# GPU 지정 및 경로 설정
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1번 사용

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # sys.path에 상위 디렉토리를 추가하여 모듈 임포트 경로 설정

import torch
import random
import numpy as np
# 1. Random seed 고정
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 2. CUDNN 비결정성 비활성화
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 라이브러리 임포트 (pytorch, 데이터 처리 등)
import yaml  # 환경설정
import argparse
import clip

from torchvision import transforms  # 이미지 변환
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# Rabbit1 모델 및 유틸 모듈 import
from models.backbone import SimSiamBackbone  # ResNet-50 기반 네트워크 백본 클래스
from features.embeddings import compute_embedding  # 이미지를 받아 백본 모델의 임베딩 출력을 얻는 함수
from features.generic_feature_bank import build_gfb
from models.gradcam import GradCAM
from utils.similarity_utils1_2 import generate_patch_based_target
from utils.image_utils import (
    load_and_preprocess_image,  # 이미지 로드 및 전처리
    overlay_heatmap,
    assemble_2x2_grid,
    save_image
)



# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


# main.py 설정 로드 및 디바이스 설정
def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)


def extract_reference_embeddings(model, config, transform):
    # main.py 참조(reference) 이미지 임베딩 불러오기
    image_dir = config['data']['reference_dir']  # e.g., "data/rabbits" 참조 이미지 폴더 지정
    save_path = config['features']['embedding_cache']
    # e.g., "features/embeddings.pth" 임베딩 캐시 로드 (키=파일명, 값=해당 이미지의 256차원 임베딩 텐서)
    # extract_features.py를 통해 모든 기준 이미지의 임베딩을 한 번에 계산/저장해두고 활용 - 매 테스트마다 기준 이미지를 일일이 모델에 넣지 않고 바로 임베딩 비교 (속도 up)

    embeddings = {}
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        path = os.path.join(image_dir, fname)
        image = load_and_preprocess_image(path, transform).unsqueeze(0).to(device)
        emb = compute_embedding(model, image)
        embeddings[fname] = emb.squeeze(0).cpu()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(embeddings, save_path)
    print(f"[✓] Saved reference embeddings to {save_path}")
    return embeddings


def retrieve_top1_clipfiltered(test_img_path, model, reference_dict, config, transform):
    """
    SimSiam cosine similarity top-5 → 그 중 CLIPScore 최고 후보 반환
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    names = list(reference_dict.keys())
    mat = torch.stack([reference_dict[n] for n in names])
    test_img = load_and_preprocess_image(test_img_path, transform).unsqueeze(0).to(device)
    test_emb = compute_embedding(model, test_img, no_grad=True)

    sims = cosine_similarity(test_emb.cpu().numpy(), mat.numpy())  # [1, N]
    idxs = sims[0].argsort()[::-1][:5]
    top5_names = [names[i] for i in idxs]

    def clip_score(img1_path, img2_path):
        image1 = preprocess(Image.open(img1_path)).unsqueeze(0).to(device)
        image2 = preprocess(Image.open(img2_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb1 = clip_model.encode_image(image1).float()
            emb2 = clip_model.encode_image(image2).float()
            emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
            emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
        return (emb1 * emb2).sum().item()

    best_score = -1
    best_name = None
    test_img_abs = os.path.abspath(test_img_path)
    ref_dir = config['data']['reference_dir']

    for name in top5_names:
        ref_path = os.path.join(ref_dir, name)
        score = clip_score(test_img_abs, ref_path)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name, best_score


def run_pipeline(test_img_path, gfb_option='A'):
    config = load_config()

    # --------------------------
    # 1. Model & Transform
    # --------------------------
    model = SimSiamBackbone(pretrained=True).to(device)  # 사전학습된 SimSiam 백본 모델 초기화
    '''
    Grad_CAM_Siamese 원본의 시암쌍 네트워크에서 파생되었지만 
        (원본 모델의 SiameseNetwork는 ResNet-50 기반 양분망으로 두 입력 이미지 각각을 Conv 통과 
        -> FC 통해 2차원 출력(거리 및 유사도 확률)을 산출
        contrastive Loss 학습을 위해 output이 sigmoid로 "같은/다른 클래스" 확률을 내는 구조였음)

    두 이미지 입력 대산 단일 이미지 임베딩 추출용으로 사용
    (원본 모델은 두 이미지를 받아 거리(Dw)와 시밀러리티(sigmoid 출력)를 내놓았으나,
    Rabbit1에서는 마지막 시밀러리티 예측 레이어를 제외하고 256차원 임베딩 추출까지만 사용 = 모델을 특징 추출기(feature extractor)로 활용) 
        (Rabbit1에서는 모든 이미지가 한 클래스(토끼)로, 대조학습이 불필요
        FC 출력층과 contrastive 손실 부분은 제거하고 256차원 임베딩만 재사용 - 두 이미지의 임베딩 사이 코사인 유사도를 계산하는 방식으로 전환)
    -> 모델 forward를 한 이미지씩 호출 - 결과 임베딩을 가지고 별도로 유사도를 구함
    '''
    model.eval()  # 평가모드로 전환

    transform = transforms.Compose([
        transforms.Resize(config['image']['size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['image']['normalization']['mean'],
                             std=config['image']['normalization']['std'])
    ])


    # --------------------------
    # 2. Extract embeddings
    # --------------------------

    reference_embeddings = extract_reference_embeddings(model, config, transform)

    # --------------------------
    # 3. Build GFB
    # --------------------------
    gfb_path = config['features']['generic_bank']
    patch_file = config['features']['patch_file']
    from scripts.extract_patches import extract_all_patches
    if not os.path.exists(patch_file):
        print(f"[!] Patch file not found: {patch_file}")
        print(f"[→] Extracting patches using conv5 feature maps...")
        extract_all_patches(model, config, transform, device)
    gfb_tensor = build_gfb(patch_file, save_path=gfb_path, option=gfb_option)
    gfb_tensor = gfb_tensor.to(device)

    # --------------------------
    # 4. Retrieve Top-1
    # --------------------------
    img_test = load_and_preprocess_image(test_img_path, transform).unsqueeze(0).to(device)
    emb_test = compute_embedding(model, img_test, no_grad=True)  # retrieval only
    top1_name, sim_score = retrieve_top1_clipfiltered(test_img_path, model, reference_embeddings, config, transform)

    top1_path = os.path.join(config['data']['reference_dir'], top1_name)
    img_ref = load_and_preprocess_image(top1_path, transform).unsqueeze(0).to(device)
    emb_ref = compute_embedding(model, img_ref, no_grad=True)

    print(f"[✓] Top-1 for {os.path.basename(test_img_path)} → {top1_name}  ({sim_score:.4f})")

    # --------------------------
    # 5. Grad-CAM + GFB 시각화
    # --------------------------
    gradcam = GradCAM(model, ['encoder.7'])



    def get_masked_cam(query_tensor, target_tensor, fmap):
        # 1. backbone → feature map → projection
        feat = model.forward_backbone(query_tensor)  # [1, 2048, H, W]
        pooled = F.adaptive_avg_pool2d(feat, (1, 1))  # [1, 2048, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [1, 2048]
        pooled.requires_grad_(True)

        z1 = model.projector(pooled)  # [1, 256]
        z2 = target_tensor.detach()  # [1, 256] projection-based target

        # 2. cosine similarity
        sim = F.cosine_similarity(z1, z2, dim=1)  # [1]
        target_score = sim.sum()

        # 3. Grad-CAM
        cam = gradcam.generate(query_tensor, target_score)[0]

        # 4. GFB 마스킹
        return filter_with_gfb(cam, fmap, gfb_tensor, threshold = config['gfb']['threshold'])

    # Grad-CAM 맵 (기존 방식 유지)
    gradcam = GradCAM(model, ['encoder.7'])

    z_test = compute_embedding(model, img_test, no_grad=False)
    z_ref = compute_embedding(model, img_ref, no_grad=False)

    raw_test = Image.open(test_img_path).convert('RGB').resize(config['image']['size'])
    raw_ref = Image.open(top1_path).convert('RGB').resize(config['image']['size'])

    fmap_test = model.get_feature_map(img_test).squeeze(0)
    fmap_ref = model.get_feature_map(img_ref).squeeze(0)

    # ---------- 1. Factual (No GFB) Grad-CAM Only ---------- #

    # [Test → Ref]
    feat_test = model.forward_backbone(img_test)  # [1, 2048, H, W]
    pooled_test = F.adaptive_avg_pool2d(feat_test, (1, 1)).view(1, -1).requires_grad_()  # [1, 2048] + grad
    z_test_proj = model.projector(pooled_test)  # [1, 256]

    feat_ref = model.forward_backbone(img_ref)  # [1, 2048, H, W]
    pooled_ref = F.adaptive_avg_pool2d(feat_ref, (1, 1)).view(1, -1).detach()  # [1, 2048] + no grad
    z_ref_proj = model.projector(pooled_ref)  # [1, 256]

    score_test_fg = F.cosine_similarity(z_test_proj, z_ref_proj, dim=1).sum()
    cam0_test = gradcam.generate(img_test, score_test_fg)[0]

    # [Ref → Test]
    # ---------- Ref → Test (Factual, No GFB) ---------- #
    feat_ref = model.forward_backbone(img_ref)
    feat_ref.requires_grad_()  # 🔥 역전파 연결
    feat_ref.retain_grad()  # 🔥 .grad가 None이 되지 않게

    pooled_ref = F.adaptive_avg_pool2d(feat_ref, (1, 1)).view(1, -1)
    z_ref_proj = model.projector(pooled_ref)

    feat_test = model.forward_backbone(img_test)
    pooled_test = F.adaptive_avg_pool2d(feat_test, (1, 1)).view(1, -1).detach()
    z_test_proj = model.projector(pooled_test)

    score = F.cosine_similarity(z_ref_proj, z_test_proj, dim=1).sum()
    score.backward(retain_graph=True)

    # 디버깅
    print(f"[CHECK] feat_ref.grad mean: {feat_ref.grad.mean().item():.6f}")

    # Grad-CAM 실행
    cam0_ref = gradcam.generate_from_features(feat_ref, feat_ref.grad, img_ref)

    # Visualization
    vis0 = overlay_heatmap(raw_test, cam0_test)
    vis1 = overlay_heatmap(raw_ref, cam0_ref)

    # ---------- 2. factual (GFB 필터 적용) ---------- #
    emb_ref_proj = model.projector(emb_ref.to(device))  # [1, 256]
    cam1 = get_masked_cam(img_test, emb_ref_proj, fmap_test)
    emb_test_proj = model.projector(emb_test.to(device))  # [1, 256]
    cam2 = get_masked_cam(img_ref, emb_test_proj, fmap_ref)
    vis2 = overlay_heatmap(raw_test, cam1)
    vis3 = overlay_heatmap(raw_ref, cam2)

    # ---------- 3. counterfactual (patch-aware Grad-CAM) ---------- #
    from utils.similarity_utils1_2 import generate_patch_based_target

    # 1. patch-aware target weights
    target_weights = generate_patch_based_target(
        fmap_test, fmap_ref, gfb_tensor,
        threshold=config['gfb']['threshold'],
        gfb_chunk_size=32,
        ref_chunk_size=128
    )  # [H, W]

    # 2. Get conv5 feature map from backbone
    feat_test = model.forward_backbone(img_test)  # [1, 2048, H, W]
    feat_test.retain_grad()

    # Grad-CAM target으로 직접 projection feature를 선택해서
    # test에서 counterfactual한 patch 위치의 projection vector 하나를 선택해서 score로 씀
    with torch.no_grad():
        cf_score_map = target_weights
        max_y, max_x = torch.where(cf_score_map == cf_score_map.max())
        y, x = max_y[0].item(), max_x[0].item()

    # 1. Grad-CAM hook 기준 feature map
    feat_test = model.forward_backbone(img_test).detach()  # [1, 2048, H, W]

    # 2. 해당 위치의 vector → projection → scalar
    patch_vector = feat_test[0, :, y, x].unsqueeze(0).requires_grad_()  # [1, 2048]
    proj_vector = model.projector(patch_vector)  # [1, 256]
    score_test_cf = proj_vector.sum()

    # 3. Grad-CAM 수행
    cam_test_cf = gradcam.generate(img_test, score_test_cf)[0]

    # 4. ref image: 기존 방식 (유사도 음수)
    z_ref_cf = model.get_embedding(img_ref).requires_grad_()
    z_test_cf = model.get_embedding(img_test).detach()
    score_ref_cf = -F.cosine_similarity(z_ref_cf, z_test_cf, dim=1).sum()
    cam_ref_cf = gradcam.generate(img_ref, score_ref_cf)[0]

    # 5. counterfactual 마스크 (GFB 기반 filtering)
    from utils.similarity_utils import compute_counterfactual_score
    cf_mask_test = compute_counterfactual_score(fmap_test, fmap_ref, gfb_tensor, threshold=config['gfb']['threshold'])
    cf_mask_ref = compute_counterfactual_score(fmap_ref, fmap_test, gfb_tensor, threshold=config['gfb']['threshold'])

    cf_mask_test_up = F.interpolate(cf_mask_test.unsqueeze(0).unsqueeze(0), size=cam_test_cf.shape, mode='bilinear',
                                    align_corners=False).squeeze()
    cf_mask_ref_up = F.interpolate(cf_mask_ref.unsqueeze(0).unsqueeze(0), size=cam_ref_cf.shape, mode='bilinear',
                                   align_corners=False).squeeze()

    # 6. masking & visualization
    cam3 = cam_test_cf * cf_mask_test_up.cpu().numpy()
    cam4 = cam_ref_cf * cf_mask_ref_up.cpu().numpy()
    vis4 = overlay_heatmap(raw_test, cam3)
    vis5 = overlay_heatmap(raw_ref, cam4)

    labels = [
        "Test - Factual (No GFB)", "Ref - Factual (No GFB)",
        "Test - Factual (GFB)", "Ref - Factual (GFB)",
        "Test - Counterfactual", "Ref - Counterfactual"
    ]

    grid = assemble_2x2_grid(
        [vis0, vis1, vis2, vis3, vis4, vis5],
        labels=labels,
        rows=3, cols=2)

    os.makedirs('output', exist_ok=True)
    fname = os.path.splitext(os.path.basename(test_img_path))[0]
    save_image(grid, f'output/main1_2_{fname}_explanation_full.png')
    print(f"[✓] Saved 2x3 explanation to output/main1_2_{fname}_explanation_full.png")


def generate_gradcam_heatmap(model, gradcam, input_tensor, target_tensor, emphasize=True):
    """
    Compute Grad-CAM heatmap given query and target tensor.
    Both must have gradient enabled (i.e., from get_embedding, not no_grad).

    If emphasize=True, apply softmax-like nonlinear transformation to boost salient regions.
    """
    # 1. Cosine similarity → scalar score
    sim = F.cosine_similarity(input_tensor, target_tensor, dim=1)  # [1]
    score = sim.sum()  # scalar

    # 2. Grad-CAM heatmap
    cam = gradcam.generate(input_tensor=input_tensor, target_score=score)[0]  # numpy [H, W]

    # 3. Emphasize salient regions using exp
    if emphasize:
        import numpy as np
        cam = np.clip(cam, 0, None)  # 음수 제거
        cam = np.exp(3 * cam)  # 3배 강화 후 exp
        cam = cam / (np.max(cam) + 1e-8)  # 정규화

    return cam


def filter_with_gfb(heatmap, fmap, gfb, threshold=None):
    if threshold is None:
        config = load_config()
        threshold = config['gfb']['threshold']

    C, H, W = fmap.shape
    mask = torch.zeros((H, W), dtype=torch.float32, device=fmap.device)

    for y in range(H):
        for x in range(W):
            patch = fmap[:, y, x]
            '''
            sims = F.cosine_similarity(patch.unsqueeze(0), gfb)
            if sims.max() < threshold:
                mask[y, x] = 1.0
            '''
            # ----------------------------------
            sims = F.cosine_similarity(patch.unsqueeze(0), gfb)  # [N]
            topk = torch.topk(sims, k=5).values  # 상위 5개 유사도
            # if topk.mean() < threshold:
            ##################################
            if topk.mean() < threshold and topk.max() < threshold + 0.05:
            #################################
                mask[y, x] = 1.0
            # ----------------------------------

    #  업샘플링 (heatmap 크기와 일치시키기)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=heatmap.shape, mode='bilinear',
                         align_corners=False)
    mask = mask.squeeze().cpu().numpy()

    return heatmap * mask




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img", type=str, required=True, help="Path to test image")
    parser.add_argument("--gfb_option", type=str, default='A', choices=['A', 'B'], help="GFB 생성 옵션")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_pipeline(args.test_img, args.gfb_option)