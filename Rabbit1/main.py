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
from sklearn.metrics.pairwise import cosine_similarity

from models.backbone import SimSiamBackbone
from models.patch_sim_predictor import PatchSimPredictor
from features.generic_feature_bank import build_gfb
from utils.image_utils import (
    load_and_preprocess_image,
    overlay_heatmap,
    assemble_2x2_grid,
    save_image
)
from utils.similarity_utils import compute_patchwise_similarity


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
        emb = model.get_embedding(image)
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


def filter_with_gfb(sim_map, fmap, gfb_tensor, threshold=0.9, output_size=(512, 512)):
    """
    Generic Feature Bank 기반 마스킹
    """
    C, H, W = fmap.shape
    mask = torch.zeros((1, H, W), device=fmap.device)

    for y in range(H):
        for x in range(W):
            patch = fmap[:, y, x]
            sims = F.cosine_similarity(patch.unsqueeze(0), gfb_tensor)  # [K]
            if sims.max() < threshold:
                mask[0, y, x] = 1.0

    sim_map = sim_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    sim_map_up = F.interpolate(sim_map, size=output_size, mode='bilinear', align_corners=False)
    mask_up = F.interpolate(mask.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False)

    result = (sim_map_up * mask_up).squeeze().detach().cpu().numpy()
    return result


def run_pipeline(test_img_path):
    config = load_config()

    # 1. Load models
    model = SimSiamBackbone(pretrained=True).to(device)
    model.eval()

    sim_model = PatchSimPredictor().to(device)
    sim_model.load_state_dict(torch.load("models/patch_sim_predictor.pth", map_location=device))
    sim_model.eval()

    # 2. Load transform
    transform = transforms.Compose([
        transforms.Resize(config['image']['size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['image']['normalization']['mean'],
                             std=config['image']['normalization']['std'])
    ])

    # 3. Extract reference embeddings
    reference_embeddings = extract_reference_embeddings(model, config, transform)

    # 4. Retrieve top-1
    img_test = load_and_preprocess_image(test_img_path, transform).unsqueeze(0).to(device)
    emb_test = model.get_embedding(img_test)
    top1_name, sim_score = retrieve_top1(emb_test, reference_embeddings)

    top1_path = os.path.join(config['data']['reference_dir'], top1_name)
    img_ref = load_and_preprocess_image(top1_path, transform).unsqueeze(0).to(device)

    print(f"[✓] Top-1 for {os.path.basename(test_img_path)} → {top1_name}  ({sim_score:.4f})")

    # 5. Build GFB
    gfb_path = config['features']['generic_bank']
    patch_file = config['features']['patch_file']
    from scripts.extract_patches import extract_all_patches
    if not os.path.exists(patch_file):
        print(f"[!] Patch file not found: {patch_file}")
        print(f"[→] Extracting patches using conv5 feature maps...")
        extract_all_patches(model, config, transform, device)
    gfb_tensor = build_gfb(patch_file, save_path=gfb_path, option='A').to(device)

    # 6. Patch-wise similarity
    fmap_test = model.get_feature_map(img_test).squeeze(0)  # [C, H, W]
    fmap_ref = model.get_feature_map(img_ref).squeeze(0)

    sim_map_1 = compute_patchwise_similarity(fmap_test, fmap_ref, sim_model)      # Test - Factual
    sim_map_2 = compute_patchwise_similarity(fmap_ref, fmap_test, sim_model)      # Ref - Factual
    sim_map_3 = compute_patchwise_similarity(fmap_test, -fmap_ref, sim_model)     # Test - Counter
    sim_map_4 = compute_patchwise_similarity(fmap_ref, -fmap_test, sim_model)     # Ref - Counter

    # 7. GFB 마스킹 및 업샘플링
    sim_map_1 = filter_with_gfb(sim_map_1, fmap_test, gfb_tensor)
    sim_map_2 = filter_with_gfb(sim_map_2, fmap_ref, gfb_tensor)
    sim_map_3 = filter_with_gfb(sim_map_3, fmap_test, gfb_tensor)
    sim_map_4 = filter_with_gfb(sim_map_4, fmap_ref, gfb_tensor)

    # 8. Load raw images
    raw_test = Image.open(test_img_path).convert('RGB').resize(config['image']['size'])
    raw_ref = Image.open(top1_path).convert('RGB').resize(config['image']['size'])

    # 9. Overlay
    vis1 = overlay_heatmap(raw_test, sim_map_1)
    vis2 = overlay_heatmap(raw_ref,  sim_map_2)
    vis3 = overlay_heatmap(raw_test, sim_map_3)
    vis4 = overlay_heatmap(raw_ref,  sim_map_4)

    grid = assemble_2x2_grid([vis1, vis2, vis3, vis4], labels=[
        "Test - Factual", "Reference - Factual",
        "Test - Counterfactual", "Reference - Counterfactual"
    ])

    os.makedirs('output', exist_ok=True)
    fname = os.path.splitext(os.path.basename(test_img_path))[0]
    save_path = f'output/{fname}_cce_explanation.png'
    save_image(grid, save_path)
    print(f"[✓] Saved CCE explanation to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_pipeline(args.test_img)
