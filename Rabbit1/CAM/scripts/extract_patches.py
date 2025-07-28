import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# scripts/extract_patches.py

import os
import torch
import yaml
from torchvision import transforms
from tqdm import tqdm

from models.backbone import SimSiamBackbone
from utils.image_utils import load_and_preprocess_image

def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def extract_all_patches(model, config, transform, device):
    image_dir = config['data']['reference_dir']
    save_path = config['features']['patch_file']

    all_patches = []

    for fname in tqdm(sorted(os.listdir(image_dir))):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        path = os.path.join(image_dir, fname)
        image = load_and_preprocess_image(path, transform).unsqueeze(0).to(device)

        with torch.no_grad():
            fmap = model.get_feature_map(image).squeeze(0)  # [C, 7, 7]
            C, H, W = fmap.shape
            patches = fmap.permute(1, 2, 0).reshape(H * W, C)  # [49, C]
            all_patches.append(patches)

    all_patches_tensor = torch.cat(all_patches, dim=0)  # [N_total, C]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(all_patches_tensor, save_path)
    print(f"[✓] Saved patch features to {save_path}")

if __name__ == "__main__":
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimSiamBackbone(pretrained=True).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(config['image']['size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['image']['normalization']['mean'],
                             std=config['image']['normalization']['std'])
    ])

    extract_all_patches(model, config, transform)
