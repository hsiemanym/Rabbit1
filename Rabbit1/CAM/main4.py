# main4.py

# GPU 지정 및 경로 설정
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1번 사용

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

from torchvision import transforms  # 이미지 변환
from PIL import Image, ImageDraw, ImageFont
import clip
import torch.nn.functional as F

# Rabbit1 모델 및 유틸 모듈 import
from utils.similarity_utils1_2 import generate_patch_based_target, compute_counterfactual_score
from models.backbone import SimSiamBackbone  # ResNet-50 기반 네트워크 백본 클래스
from features.embeddings import compute_embedding  # 이미지를 받아 백본 모델의 임베딩 출력을 얻는 함수
from features.generic_feature_bank import build_gfb
from models.gradcam import GradCAM

from utils.image_utils import (
    load_and_preprocess_image,  # 이미지 로드 및 전처리
    overlay_heatmap,
    assemble_2x2_grid,
    save_image
)

from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


# main.py 설정 로드 및 디바이스 설정
def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)


def extract_reference_embeddings(model, config, transform):
    image_dir = config['data']['reference_dir']
    save_path = config['features']['embedding_cache']

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


# --------------------------------------------------------------------------------------------
# [수정된 부분]
# Factual(NoGFB) 히트맵을 생성하기 위한 새로운 함수 추가
# --------------------------------------------------------------------------------------------
'''def compute_factual_score_by_masking(model, img_test, img_ref):
    """
    Test 이미지의 패치를 하나씩 마스킹하며 유사도 변화를 측정하여 Factual 히트맵 생성
    """
    with torch.no_grad():
        fmap_test = model.get_feature_map(img_test).squeeze(0)  # [C, H, W]
        fmap_ref = model.get_feature_map(img_ref).squeeze(0)  # [C, H, W]
        C, H, W = fmap_test.shape

        # 전체 이미지 유사도를 baseline으로 설정
        # 텐서 형태를 [1, 2048]로 변경
        emb_test_orig = model.projector(F.adaptive_avg_pool2d(fmap_test.unsqueeze(0), (1, 1)).view(1, -1))
        emb_ref_orig = model.projector(F.adaptive_avg_pool2d(fmap_ref.unsqueeze(0), (1, 1)).view(1, -1))
        original_sim = F.cosine_similarity(emb_test_orig, emb_ref_orig, dim=1).item()

        heatmap = torch.zeros((H, W), device=fmap_test.device)

        # 패치 단위로 마스킹 후 유사도 변화 측정
        for y in range(H):
            for x in range(W):
                fmap_masked = fmap_test.clone()
                fmap_masked[:, y, x] = 0

                # 마스킹된 feature map으로 유사도 계산
                # 텐서 형태를 [1, 2048]로 변경
                emb_masked = model.projector(F.adaptive_avg_pool2d(fmap_masked.unsqueeze(0), (1, 1)).view(1, -1))
                masked_sim = F.cosine_similarity(emb_masked, emb_ref_orig, dim=1).item()

                score = original_sim - masked_sim
                heatmap[y, x] = score

        heatmap = F.relu(heatmap)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap.cpu().numpy()
'''


# Factual(NoGFB) 히트맵을 생성하기 위한 새로운 Grad-CAM 기반 함수
# --------------------------------------------------------------------------------------------
def get_factual_gradcam(model, gradcam, query_tensor, target_emb):
    """
    쿼리 이미지와 타겟 임베딩 사이의 유사성을 기반으로 Grad-CAM 히트맵을 생성합니다.
    """
    # 쿼리 이미지 임베딩 추출
    query_emb = model.get_embedding(query_tensor)

    # 유사도 계산 (target_emb와 gradient를 연결)
    similarity = F.cosine_similarity(query_emb, target_emb.detach(), dim=1)

    # Grad-CAM은 스칼라 값을 타겟으로 함
    target_score = similarity.sum()

    # Grad-CAM 생성
    cam = gradcam.generate(input_tensor=query_tensor, target_score=target_score)[0]
    return cam


# --------------------------------------------------------------------------------------------


# main4.py (전체 파일에서 run_pipeline 함수를 아래 코드로 교체)

def run_pipeline(test_img_path, gfb_option='A'):
    config = load_config()
    model = SimSiamBackbone(pretrained=True).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(config['image']['size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['image']['normalization']['mean'],
                             std=config['image']['normalization']['std'])
    ])

    reference_embeddings = extract_reference_embeddings(model, config, transform)

    gfb_path = config['features']['generic_bank']
    patch_file = config['features']['patch_file']
    from scripts.extract_patches import extract_all_patches
    if not os.path.exists(patch_file):
        print(f"[!] Patch file not found: {patch_file}")
        print(f"[→] Extracting patches using conv5 feature maps...")
        extract_all_patches(model, config, transform, device)
    gfb_tensor = build_gfb(patch_file, save_path=gfb_path, option=gfb_option)
    gfb_tensor = gfb_tensor.to(device)

    img_test = load_and_preprocess_image(test_img_path, transform).unsqueeze(0).to(device)
    top1_name, sim_score = retrieve_top1_clipfiltered(test_img_path, model, reference_embeddings, config, transform)

    top1_path = os.path.join(config['data']['reference_dir'], top1_name)
    img_ref = load_and_preprocess_image(top1_path, transform).unsqueeze(0).to(device)

    # Factual (No GFB)와 Factual (GFB) 모두에 사용할 256차원 임베딩 추출
    emb_test_proj = model.get_embedding(img_test)
    emb_ref_proj = model.get_embedding(img_ref)

    print(f"[✓] Top-1 for {os.path.basename(test_img_path)} → {top1_name}  ({sim_score:.4f})")

    gradcam = GradCAM(model, ['encoder.7'])

    def get_masked_cam(query_tensor, target_tensor, fmap):
        feat = model.forward_backbone(query_tensor)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1))
        pooled = pooled.view(pooled.size(0), -1)
        pooled.requires_grad_(True)
        z1 = model.projector(pooled)
        z2 = target_tensor.detach()
        sim = F.cosine_similarity(z1, z2, dim=1)
        target_score = sim.sum()
        cam = gradcam.generate(query_tensor, target_score)[0]
        return filter_with_gfb(cam, fmap, gfb_tensor)

    raw_test = Image.open(test_img_path).convert('RGB').resize(config['image']['size'])
    raw_ref = Image.open(top1_path).convert('RGB').resize(config['image']['size'])

    fmap_test = model.get_feature_map(img_test).squeeze(0)
    fmap_ref = model.get_feature_map(img_ref).squeeze(0)

    # --------------------------
    # [수정된 부분] 1. Factual (No GFB)
    # projector를 우회하여 pooled feature map의 유사도를 타겟으로 사용
    # --------------------------
    def generate_factual_cam(query_img, target_img):
        feat_query = model.forward_backbone(query_img)
        feat_target = model.forward_backbone(target_img).detach()

        pooled_query = F.adaptive_avg_pool2d(feat_query, (1, 1)).view(1, -1)
        pooled_target = F.adaptive_avg_pool2d(feat_target, (1, 1)).view(1, -1)

        similarity = F.cosine_similarity(pooled_query, pooled_target, dim=1)
        target_score = similarity.sum()

        cam = gradcam.generate(query_img, target_score)[0]
        return cam

    cam0_ref = generate_factual_cam(img_test, img_ref)
    cam0_test = generate_factual_cam(img_ref, img_test)

    vis0 = overlay_heatmap(raw_test, cam0_test)
    vis1 = overlay_heatmap(raw_ref, cam0_ref)

    # ---------- 2. factual (GFB 필터 적용) ---------- #
    cam1 = get_masked_cam(img_test, emb_ref_proj, fmap_test)
    cam2 = get_masked_cam(img_ref, emb_test_proj, fmap_ref)
    vis2 = overlay_heatmap(raw_test, cam1)
    vis3 = overlay_heatmap(raw_ref, cam2)

    # ---------- 3. counterfactual (patch-aware Grad-CAM) ---------- #
    target_weights = generate_patch_based_target(
        fmap_test, fmap_ref, gfb_tensor,
        threshold=config['gfb']['threshold'],
        gfb_chunk_size=32,
        ref_chunk_size=128
    )
    feat_test = model.forward_backbone(img_test)
    feat_test.retain_grad()

    with torch.no_grad():
        cf_score_map = target_weights
        max_y, max_x = torch.where(cf_score_map == cf_score_map.max())
        y, x = max_y[0].item(), max_x[0].item()

    feat_test = model.forward_backbone(img_test)
    feat_test.retain_grad()

    patch_vector = feat_test[0, :, y, x].unsqueeze(0)
    proj_vector = model.projector(patch_vector)
    score_test_cf = proj_vector.sum()

    score_test_cf.backward(retain_graph=True)
    cam_test_cf = gradcam.generate_from_features(feat_test, feat_test.grad, img_test)

    z_ref_cf = model.get_embedding(img_ref).requires_grad_()
    z_test_cf = model.get_embedding(img_test).detach()
    score_ref_cf = -F.cosine_similarity(z_ref_cf, z_test_cf, dim=1).sum()
    cam_ref_cf = gradcam.generate(img_ref, score_ref_cf)[0]

    from utils.similarity_utils import compute_counterfactual_score
    cf_mask_test = compute_counterfactual_score(fmap_test, fmap_ref, gfb_tensor, threshold=config['gfb']['threshold'])
    cf_mask_ref = compute_counterfactual_score(fmap_ref, fmap_test, gfb_tensor, threshold=config['gfb']['threshold'])

    cf_mask_test_up = F.interpolate(cf_mask_test.unsqueeze(0).unsqueeze(0), size=cam_test_cf.shape, mode='bilinear',
                                    align_corners=False).squeeze()
    cf_mask_ref_up = F.interpolate(cf_mask_ref.unsqueeze(0).unsqueeze(0), size=cam_ref_cf.shape, mode='bilinear',
                                   align_corners=False).squeeze()

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
    save_image(grid, f'output/main4_{fname}_explanation_full.png')
    print(f"[✓] Saved 2x3 explanation to output/main4_{fname}_explanation_full.png")


def get_factual_gradcam(model, gradcam, query_tensor, target_emb):
    """
    쿼리 이미지와 타겟 임베딩 사이의 유사성을 기반으로 Grad-CAM 히트맵을 생성합니다.
    """
    # 쿼리 이미지 임베딩 추출
    query_emb = model.get_embedding(query_tensor)

    # 유사도 계산 (target_emb와 gradient를 연결)
    similarity = F.cosine_similarity(query_emb, target_emb.detach(), dim=1)

    # Grad-CAM은 스칼라 값을 타겟으로 함
    target_score = similarity.sum()

    # Grad-CAM 생성
    cam = gradcam.generate(input_tensor=query_tensor, target_score=target_score)[0]
    return cam


def generate_gradcam_heatmap(model, gradcam, input_tensor, target_tensor):
    sim = F.cosine_similarity(input_tensor, target_tensor, dim=1)
    score = sim.sum()
    cam = gradcam.generate(input_tensor=input_tensor, target_score=score)[0]
    return cam


def filter_with_gfb(heatmap, fmap, gfb, threshold=0.7):
    C, H, W = fmap.shape
    mask = torch.zeros((H, W), dtype=torch.float32, device=fmap.device)

    for y in range(H):
        for x in range(W):
            patch = fmap[:, y, x]
            sims = F.cosine_similarity(patch.unsqueeze(0), gfb)
            topk = torch.topk(sims, k=5).values
            if topk.mean() < threshold:
                mask[y, x] = 1.0

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