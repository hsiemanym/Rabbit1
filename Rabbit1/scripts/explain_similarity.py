import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# scripts/explain_similarity.py

import os
import torch
import yaml
import argparse
from torchvision import transforms
from PIL import Image

from models.backbone import SimSiamBackbone
from models.patch_sim_predictor import PatchSimilarityPredictor
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


def explain_similarity(test_path, ref_path, model, sim_model, config, device):
    transform = transforms.Compose([
        transforms.Resize(config['image']['size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['image']['normalization']['mean'],
                             std=config['image']['normalization']['std'])
    ])

    # Load images
    img1 = load_and_preprocess_image(test_path, transform).unsqueeze(0).to(device)
    img2 = load_and_preprocess_image(ref_path, transform).unsqueeze(0).to(device)

    fmap1 = model.get_feature_map(img1).squeeze(0)  # [C, H, W]
    fmap2 = model.get_feature_map(img2).squeeze(0)

    sim_map = compute_patchwise_similarity(fmap1, fmap2, sim_model).detach().cpu().numpy()  # [H, W]
    sim_map_cf = compute_patchwise_similarity(fmap1, -fmap2, sim_model).detach().cpu().numpy()

    raw1 = Image.open(test_path).convert("RGB").resize(config['image']['size'])
    raw2 = Image.open(ref_path).convert("RGB").resize(config['image']['size'])

    vis1 = overlay_heatmap(raw1, sim_map)
    vis2 = overlay_heatmap(raw2, sim_map)
    vis3 = overlay_heatmap(raw1, sim_map_cf)
    vis4 = overlay_heatmap(raw2, sim_map_cf)

    labels = ["Test - Factual", "Reference - Factual", "Test - Counterfactual", "Reference - Counterfactual"]
    grid = assemble_2x2_grid([vis1, vis2, vis3, vis4], labels=labels)

    os.makedirs("output", exist_ok=True)
    fname = os.path.splitext(os.path.basename(test_path))[0]
    save_path = f"output/{fname}_cce_explanation.png"
    save_image(grid, save_path)
    print(f"[✓] Saved CCE explanation to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img", type=str, required=True)
    parser.add_argument("--ref_img", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()

    model = SimSiamBackbone(pretrained=True).to(device)
    model.eval()

    sim_model = PatchSimilarityPredictor().to(device)
    sim_model.load_state_dict(torch.load("models/patch_sim_predictor.pth", map_location=device))
    sim_model.eval()

    explain_similarity(args.test_img, args.ref_img, model, sim_model, config, device)
