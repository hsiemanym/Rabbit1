# main4.py

# GPU 지정 및 경로 설정
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import random
import numpy as np
import argparse
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

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

# --- 새로 분리한 모듈 import ---
from utils.config_utils import load_config
from features.retrieval_features import extract_reference_embeddings, retrieve_top1_clipfiltered
from features.gfb_utils import filter_with_gfb
# from utils.gradcam_utils import get_factual_gradcam, generate_gradcam_heatmap # 이 함수들은 run_pipeline 내 로컬 함수로 대체됨

# --- 기존 모듈 import ---
from models.backbone import SimSiamBackbone
from features.generic_feature_bank import build_gfb
from models.gradcam import GradCAM
from utils.image_utils import load_and_preprocess_image, overlay_heatmap, assemble_2x2_grid, save_image
from utils.similarity_utils1_2 import generate_patch_based_target, compute_counterfactual_score


# main4.py (run_pipeline 함수만 교체)

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

    reference_embeddings = extract_reference_embeddings(model, config, transform, device)

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
    top1_name, sim_score = retrieve_top1_clipfiltered(test_img_path, model, reference_embeddings, config, transform, device)

    top1_path = os.path.join(config['data']['reference_dir'], top1_name)
    img_ref = load_and_preprocess_image(top1_path, transform).unsqueeze(0).to(device)

    emb_test_proj = model.get_embedding(img_test)
    emb_ref_proj = model.get_embedding(img_ref)

    print(f"[✓] Top-1 for {os.path.basename(test_img_path)} → {top1_name}  ({sim_score:.4f})")

    gradcam = GradCAM(model, ['encoder.7'])

    raw_test = Image.open(test_img_path).convert('RGB').resize(config['image']['size'])
    raw_ref = Image.open(top1_path).convert('RGB').resize(config['image']['size'])

    fmap_test = model.get_feature_map(img_test).squeeze(0)
    fmap_ref = model.get_feature_map(img_ref).squeeze(0)

    # ---------- 1. Factual (No GFB) ----------
    def generate_factual_cam(query_img, target_img):
        feat_query = model.forward_backbone(query_img)
        feat_target = model.forward_backbone(target_img).detach()
        pooled_query = F.adaptive_avg_pool2d(feat_query, (1, 1)).view(1, -1)
        pooled_target = F.adaptive_avg_pool2d(feat_target, (1, 1)).view(1, -1)
        similarity = F.cosine_similarity(pooled_query, pooled_target, dim=1).sum()
        return gradcam.generate(query_img, similarity)[0]

    cam0_ref = generate_factual_cam(img_test, img_ref)
    cam0_test = generate_factual_cam(img_ref, img_test)
    vis0 = overlay_heatmap(raw_test, cam0_test)
    vis1 = overlay_heatmap(raw_ref, cam0_ref)

    # ---------- 2. Factual (GFB) ----------
    def get_masked_cam(query_tensor, target_tensor, fmap):
        feat = model.forward_backbone(query_tensor)
        # pooled.size(0) -> feat.size(0)으로 수정
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
        z1 = model.projector(pooled)
        z2 = target_tensor.detach()
        sim = F.cosine_similarity(z1, z2, dim=1).sum()
        cam = gradcam.generate(query_tensor, sim)[0]
        return filter_with_gfb(cam, fmap, gfb_tensor)

    cam1 = get_masked_cam(img_test, emb_ref_proj, fmap_test)
    cam2 = get_masked_cam(img_ref, emb_test_proj, fmap_ref)
    vis2 = overlay_heatmap(raw_test, cam1)
    vis3 = overlay_heatmap(raw_ref, cam2)

    # ---------- 3. Counterfactual (patch-aware Grad-CAM) ---------- #

    # Test 이미지에 대한 Counterfactual CAM 생성
    target_weights = generate_patch_based_target(
        fmap_test, fmap_ref, gfb_tensor,
        threshold=config['gfb']['threshold']
    )

    # 1. forward pass를 한 번만 실행하고 그래디언트 흐름을 설정합니다.
    feat_test = model.forward_backbone(img_test)
    feat_test.retain_grad()

    # 2. 가장 counterfactual한 패치의 위치(y, x)를 찾습니다.
    with torch.no_grad():
        max_y, max_x = torch.where(target_weights == target_weights.max())
        y, x = max_y[0].item(), max_x[0].item()

    # 3. 해당 위치의 패치 벡터를 추출하고, 이를 타겟으로 역전파를 수행합니다.
    patch_vector = feat_test[0, :, y, x].unsqueeze(0)
    score_test_cf = model.projector(patch_vector).sum()
    score_test_cf.backward(retain_graph=True)  # retain_graph=True는 ref 이미지 계산을 위해 필요합니다.

    # 4. 계산된 그래디언트를 사용해 CAM을 생성합니다.
    cam_test_cf = gradcam.generate_from_features(feat_test, feat_test.grad, img_test)

    # Reference 이미지에 대한 Counterfactual CAM 생성 (기존과 동일)
    z_ref_cf = model.get_embedding(img_ref).requires_grad_()
    z_test_cf = model.get_embedding(img_test).detach()
    score_ref_cf = -F.cosine_similarity(z_ref_cf, z_test_cf, dim=1).sum()
    cam_ref_cf = gradcam.generate(img_ref, score_ref_cf)[0]

    # 생성된 CAM에 Counterfactual 마스크를 적용
    from utils.similarity_utils import compute_counterfactual_score  # 이 import는 여기에 있어도 괜찮습니다.
    cf_mask_test = compute_counterfactual_score(fmap_test, fmap_ref, gfb_tensor, threshold=config['gfb']['threshold'])
    cf_mask_ref = compute_counterfactual_score(fmap_ref, fmap_test, gfb_tensor, threshold=config['gfb']['threshold'])

    def upscale_mask(mask, target_shape):
        return F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=target_shape, mode='bilinear',
                             align_corners=False).squeeze()

    cam3 = cam_test_cf * upscale_mask(cf_mask_test, cam_test_cf.shape).cpu().numpy()
    cam4 = cam_ref_cf * upscale_mask(cf_mask_ref, cam_ref_cf.shape).cpu().numpy()

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
    save_image(grid, f'output/main3_{fname}_explanation_full.png')
    print(f"[✓] Saved 2x3 explanation to output/main3_{fname}_explanation_full.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img", type=str, required=True, help="Path to test image")
    parser.add_argument("--gfb_option", type=str, default='A', choices=['A', 'B'], help="GFB 생성 옵션")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_pipeline(args.test_img, args.gfb_option)