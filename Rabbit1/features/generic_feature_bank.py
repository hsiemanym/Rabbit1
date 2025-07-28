# features/generic_feature_bank.py

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def build_gfb(patch_file, save_path='features/gfb_tensor.pth', option='A'):
    """
    patch_file: str, path to patch tensor saved from extract_patches
    save_path: where to save resulting GFB tensor
    option: 'A' or 'B' (not yet differentiated)
    """
    print(f"[→] Loading patches from {patch_file}")
    patch_dict = torch.load(patch_file)

    all_patches = []
    for img_patches in patch_dict.values():  # img_patches: [N, C]
        all_patches.append(img_patches)
    all_patches = torch.cat(all_patches, dim=0)  # [Total, C]

    # Normalize
    all_patches = torch.nn.functional.normalize(all_patches, dim=1)

    # Optionally cluster / reduce (for now: identity)
    gfb_tensor = all_patches

    print(f"[✓] GFB built with {gfb_tensor.shape[0]} patches.")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(gfb_tensor, save_path)
    return gfb_tensor
