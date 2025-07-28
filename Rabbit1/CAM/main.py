import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# main.py

import yaml
import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F


from models.backbone import SimSiamBackbone
from features.embeddings import compute_embedding
from features.generic_feature_bank import build_gfb
from models.gradcam import GradCAM
from utils.image_utils import (
    load_and_preprocess_image,
    overlay_heatmap,
    assemble_2x2_grid,
    save_image
)

from sklearn.metrics.pairwise import cosine_similarity


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


def retrieve_top1(test_emb, reference_dict):
    names = list(reference_dict.keys())
    mat = torch.stack([reference_dict[n] for n in names])
    sims = cosine_similarity(test_emb.cpu().numpy(), mat.numpy())  # [1, N]
    idx = sims.argmax()
    return names[idx], sims[0, idx]


def run_pipeline(test_img_path, gfb_option='A'):
    config = load_config()

    # --------------------------
    # 1. Model & Transform
    # --------------------------
    model = SimSiamBackbone(pretrained=True).to(device)
    model.eval()
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
    emb_test = compute_embedding(model, img_test)
    top1_name, sim_score = retrieve_top1(emb_test, reference_embeddings)

    top1_path = os.path.join(config['data']['reference_dir'], top1_name)
    img_ref = load_and_preprocess_image(top1_path, transform).unsqueeze(0).to(device)
    emb_ref = compute_embedding(model, img_ref)

    print(f"[✓] Top-1 for {os.path.basename(test_img_path)} → {top1_name}  ({sim_score:.4f})")

    # --------------------------
    # 5. Grad-CAM + GFB 시각화
    # --------------------------
    gradcam = GradCAM(model, ['encoder.7'])

    def get_masked_cam(query_tensor, target_tensor, fmap):
       #  query_tensor.requires_grad = True

        # 1. backbone → feature map → projection까지 그대로
        feat = model.forward_backbone(query_tensor)  # [1, C, H, W]

        pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))  # [1, 2048, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [1, 2048]
        pooled.requires_grad_(True)

        z1 = model.projector(pooled)  # [1, D]
        z2 = target_tensor.detach()  # Grad 없이 사용

        # 2. cosine similarity (backward 연결됨)
        sim = torch.nn.functional.cosine_similarity(z1, z2, dim=1)  # [1]
        target_score = sim.sum()  # scalar

        # 3. Grad-CAM → feature importance
        cam = gradcam.generate(query_tensor, target_score)[0]

        # 4. GFB 마스킹
        return filter_with_gfb(cam, fmap, gfb_tensor)

    fmap_test = model.get_feature_map(img_test).squeeze(0)
    fmap_ref = model.get_feature_map(img_ref).squeeze(0)

    # factual
    cam1 = get_masked_cam(img_test, emb_ref, fmap_test)
    cam2 = get_masked_cam(img_ref, emb_test, fmap_ref)

    # counterfactual (1 - similarity)
    inv_emb_test = -emb_test
    inv_emb_ref = -emb_ref
    cam3 = get_masked_cam(img_test, inv_emb_ref, fmap_test)
    cam4 = get_masked_cam(img_ref, inv_emb_test, fmap_ref)

    raw_test = Image.open(test_img_path).convert('RGB').resize(config['image']['size'])
    raw_ref  = Image.open(top1_path).convert('RGB').resize(config['image']['size'])

    vis1 = overlay_heatmap(raw_test, cam1)
    vis2 = overlay_heatmap(raw_ref, cam2)
    vis3 = overlay_heatmap(raw_test, cam3)
    vis4 = overlay_heatmap(raw_ref, cam4)

    labels = [
        "Test - Factual",
        "Reference - Factual",
        "Test - Counterfactual",
        "Reference - Counterfactual"
    ]

    grid = assemble_2x2_grid([vis1, vis2, vis3, vis4], labels=labels)

    os.makedirs('output', exist_ok=True)
    fname = os.path.splitext(os.path.basename(test_img_path))[0]
    save_image(grid, f'output/{fname}_explanation.png')
    print(f"[✓] Saved explanation to output/{fname}_explanation.png")


def filter_with_gfb(heatmap, fmap, gfb, threshold=0.9):
    C, H, W = fmap.shape
    mask = torch.zeros((H, W), dtype=torch.float32, device=fmap.device)

    for y in range(H):
        for x in range(W):
            patch = fmap[:, y, x]
            sims = torch.nn.functional.cosine_similarity(patch.unsqueeze(0), gfb)
            if sims.max() < threshold:
                mask[y, x] = 1.0

    #  업샘플링 (heatmap 크기와 일치시키기)
    mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=heatmap.shape, mode='bilinear', align_corners=False)
    mask = mask.squeeze().cpu().numpy()

    return heatmap * mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img", type=str, required=True, help="Path to test image")
    parser.add_argument("--gfb_option", type=str, default='A', choices=['A', 'B'], help="GFB 생성 옵션")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_pipeline(args.test_img, args.gfb_option)




