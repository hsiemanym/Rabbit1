import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from models.backbone import SimSiamBackbone
from features.generic_feature_bank import build_gfb
from models.gradcam import GradCAM
from utils.similarity_utils import compute_counterfactual_score
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
        with torch.no_grad():
            feats = model.get_feature_map(image)
            pooled = F.adaptive_avg_pool2d(feats, (1, 1)).view(1, -1)
            emb = model.projector(pooled)
        embeddings[fname] = emb.squeeze(0).cpu()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(embeddings, save_path)
    print(f"[✓] Saved reference embeddings to {save_path}")
    return embeddings


def retrieve_top1(test_emb, reference_dict):
    names = list(reference_dict.keys())
    mat = torch.stack([reference_dict[n] for n in names])
    sims = cosine_similarity(test_emb.detach().cpu().numpy(), mat.numpy())  # [1, N]
    idx = sims.argmax()
    return names[idx], sims[0, idx]


def filter_with_gfb(heatmap, fmap, gfb, threshold=0.9):
    C, H, W = fmap.shape
    mask = torch.zeros((H, W), dtype=torch.float32, device=fmap.device)

    for y in range(H):
        for x in range(W):
            patch = fmap[:, y, x]
            sims = F.cosine_similarity(patch.unsqueeze(0), gfb)
            if sims.max() < threshold:
                mask[y, x] = 1.0

    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=heatmap.shape, mode='bilinear', align_corners=False)
    return heatmap * mask.squeeze().cpu().numpy()


def run_pipeline(test_img_path, gfb_option='A'):
    config = load_config()

    # --------------------------
    # 1. Load model and transform
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
    # 2. Extract reference embeddings
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
        print(f"[→] Extracting patches...")
        extract_all_patches(model, config, transform, device)

    gfb_tensor = build_gfb(patch_file, gfb_path, option=gfb_option)
    gfb_tensor = gfb_tensor.to(device)

    # --------------------------
    # 4. Top-1 retrieval
    # --------------------------
    img_test = load_and_preprocess_image(test_img_path, transform).unsqueeze(0).to(device)
    feat_test = model.get_feature_map(img_test)
    pooled_test = F.adaptive_avg_pool2d(feat_test, (1, 1)).view(1, -1)
    emb_test = model.projector(pooled_test)
    top1_name, sim_score = retrieve_top1(emb_test, reference_embeddings)

    top1_path = os.path.join(config['data']['reference_dir'], top1_name)
    img_ref = load_and_preprocess_image(top1_path, transform).unsqueeze(0).to(device)
    feat_ref = model.get_feature_map(img_ref)
    pooled_ref = F.adaptive_avg_pool2d(feat_ref, (1, 1)).view(1, -1)
    emb_ref = model.projector(pooled_ref)

    print(f"[✓] Top-1 for {os.path.basename(test_img_path)} → {top1_name}  ({sim_score:.4f})")

    # --------------------------
    # 5. Grad-CAM + GFB (factual) and counterfactual heatmaps
    # --------------------------
    gradcam = GradCAM(model, target_layers=["encoder.7"])

    # Grad-CAM source
    pooled_test.requires_grad_(True)
    cam1 = gradcam.generate(img_test, F.cosine_similarity(model.projector(pooled_test), emb_ref).sum())[0]
    pooled_test.requires_grad_(False)

    pooled_ref.requires_grad_(True)
    cam2 = gradcam.generate(img_ref, F.cosine_similarity(model.projector(pooled_ref), emb_test).sum())[0]
    pooled_ref.requires_grad_(False)

    # factual: GFB filtering
    cam1 = filter_with_gfb(cam1, feat_test.squeeze(0), gfb_tensor)
    cam2 = filter_with_gfb(cam2, feat_ref.squeeze(0), gfb_tensor)

    # counterfactual masks
    cf_mask1 = compute_counterfactual_score(feat_test.squeeze(0), feat_ref.squeeze(0), gfb_tensor)
    cf_mask2 = compute_counterfactual_score(feat_ref.squeeze(0), feat_test.squeeze(0), gfb_tensor)

    cam3 = cam1 * cf_mask1.cpu().numpy()
    cam4 = cam2 * cf_mask2.cpu().numpy()

    # --------------------------
    # 6. Visualization
    # --------------------------
    raw_test = Image.open(test_img_path).convert('RGB').resize(config['image']['size'])
    raw_ref  = Image.open(top1_path).convert('RGB').resize(config['image']['size'])

    vis1 = overlay_heatmap(raw_test, cam1)
    vis2 = overlay_heatmap(raw_ref, cam2)
    vis3 = overlay_heatmap(raw_test, cam3)
    vis4 = overlay_heatmap(raw_ref, cam4)

    grid = assemble_2x2_grid([vis1, vis2, vis3, vis4], labels=[
        "Test - Factual", "Reference - Factual", "Test - Counterfactual", "Reference - Counterfactual"
    ])

    os.makedirs('output', exist_ok=True)
    fname = os.path.splitext(os.path.basename(test_img_path))[0]
    save_image(grid, f'output/{fname}_explanation.png')
    print(f"[✓] Saved explanation to output/{fname}_explanation.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img", type=str, required=True)
    parser.add_argument("--gfb_option", type=str, default='A', choices=['A', 'B'])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_pipeline(args.test_img, args.gfb_option)
