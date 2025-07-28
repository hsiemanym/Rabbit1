# scripts/extract_patches.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from tqdm import tqdm

from utils.image_utils import load_and_preprocess_image

def extract_all_patches(model, config, transform, device):
    """
    Extracts conv5 patches for all reference images and saves to a file.
    """
    patch_dict = {}
    image_dir = config['data']['reference_dir']
    save_path = config['features']['patch_file']

    model.eval()

    with torch.no_grad():
        for fname in tqdm(sorted(os.listdir(image_dir)), desc="Extracting patches"):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            path = os.path.join(image_dir, fname)
            image = load_and_preprocess_image(path, transform).unsqueeze(0).to(device)
            fmap = model.get_feature_map(image).squeeze(0)  # [C, H, W]
            patches = fmap.permute(1, 2, 0).reshape(-1, fmap.shape[0])  # [H*W, C]
            patch_dict[fname] = patches.cpu()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(patch_dict, save_path)
    print(f"[✓] Saved all patches to {save_path}")
