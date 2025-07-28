import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# scripts/explain_similarity.py

import os
import torch
import yaml
import argparse
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

from models.backbone import SimSiamBackbone
from models.gradcam import GradCAM
from features.embeddings import compute_embedding
from utils.image_utils import (
    load_and_preprocess_image,
    overlay_heatmap,
    assemble_2x2_grid,
    save_image,
)

# -------------------------------
# 설정 및 모델 로드
# -------------------------------
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimSiamBackbone(proj_dim=256, pretrained=True).to(device)
model.eval()

gradcam = GradCAM(model, target_layers=['encoder.7'])  # conv5 only

# transform
transform = transforms.Compose([
    transforms.Resize(config['image']['size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['image']['normalization']['mean'],
                         std=config['image']['normalization']['std'])
])

# -------------------------------
# Generic Feature Bank
# -------------------------------
GFB_PATH = config['features']['generic_bank']
generic_bank = torch.load(GFB_PATH).to(device)  # [K, D]


def is_generic_patch(patch_feat, gfb, threshold=0.9):
    sims = torch.nn.functional.cosine_similarity(patch_feat.unsqueeze(0), gfb, dim=1)
    return sims.max().item() > threshold


def filter_heatmap_with_gfb(feature_map, heatmap, gfb, threshold=0.9):
    """
    feature_map: [C, H, W]
    heatmap: [H, W]
    gfb: [K, D]
    """
    C, H, W = feature_map.shape
    mask = torch.zeros((H, W), dtype=torch.float32, device=feature_map.device)

    for y in range(H):
        for x in range(W):
            patch_vec = feature_map[:, y, x]
            if not is_generic_patch(patch_vec, gfb, threshold):
                mask[y, x] = 1.0  # non-generic → retain
    return heatmap * mask.cpu().numpy()


# -------------------------------
# 유사 이미지 검색
# -------------------------------
def retrieve_top1_embedding(test_emb, ref_emb_dict):
    names = list(ref_emb_dict.keys())
    ref_mat = torch.stack([ref_emb_dict[n] for n in names])  # [N, D]
    sims = cosine_similarity(test_emb.cpu().numpy(), ref_mat.numpy())  # [1, N]
    idx = sims.argmax()
    return names[idx], sims[0, idx]


# -------------------------------
# 실행 메인
# -------------------------------
def main(test_img_path):
    # Load reference embeddings
    ref_embeddings = torch.load(config['features']['embedding_cache'])

    # Load & embed test image
    img_test = load_and_preprocess_image(test_img_path, transform).unsqueeze(0).to(device)
    emb_test = compute_embedding(model, img_test)

    # Retrieve top-1 match
    top1_name, sim_score = retrieve_top1_embedding(emb_test, ref_embeddings)
    top1_path = os.path.join(config['data']['reference_dir'], top1_name)
    img_ref = load_and_preprocess_image(top1_path, transform).unsqueeze(0).to(device)

    print(f"[✓] Test Image: {os.path.basename(test_img_path)}")
    print(f"    Top-1 Match: {top1_name}  |  Similarity: {sim_score:.4f}")

    # Get Grad-CAMs
    factual_test = gradcam.generate(img_test, torch.tensor(sim_score, requires_grad=True))
    factual_ref = gradcam.generate(img_ref, torch.tensor(sim_score, requires_grad=True))
    counter_test = gradcam.generate(img_test, torch.tensor(1 - sim_score, requires_grad=True))
    counter_ref = gradcam.generate(img_ref, torch.tensor(1 - sim_score, requires_grad=True))

    # Get feature maps (conv5) for GFB masking
    fmap_test = model.get_feature_map(img_test).squeeze(0)  # [C, H, W]
    fmap_ref = model.get_feature_map(img_ref).squeeze(0)  # [C, H, W]

    # Filter heatmaps using GFB
    factual_test_f = filter_heatmap_with_gfb(fmap_test, factual_test[0], generic_bank)
    factual_ref_f = filter_heatmap_with_gfb(fmap_ref, factual_ref[0], generic_bank)
    counter_test_f = filter_heatmap_with_gfb(fmap_test, counter_test[0], generic_bank)
    counter_ref_f = filter_heatmap_with_gfb(fmap_ref, counter_ref[0], generic_bank)

    # Original images (PIL)
    from PIL import Image
    raw_test = Image.open(test_img_path).convert('RGB').resize(config['image']['size'])
    raw_ref = Image.open(top1_path).convert('RGB').resize(config['image']['size'])

    # Compose overlays
    vis_1 = overlay_heatmap(raw_test, factual_test_f)
    vis_2 = overlay_heatmap(raw_ref, factual_ref_f)
    vis_3 = overlay_heatmap(raw_test, counter_test_f)
    vis_4 = overlay_heatmap(raw_ref, counter_ref_f)

    final_grid = assemble_2x2_grid([vis_1, vis_2, vis_3, vis_4])

    # Save
    os.makedirs('output', exist_ok=True)
    base_name = os.path.splitext(os.path.basename(test_img_path))[0]
    save_path = f'output/{base_name}_explanation.png'
    save_image(final_grid, save_path)
    print(f"[✓] Saved explanation to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img", type=str, required=True, help="Path to test image")
    args = parser.parse_args()
    main(args.test_img)
