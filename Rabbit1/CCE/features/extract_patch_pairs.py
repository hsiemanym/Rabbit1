import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# features/extract_patch_pairs.py

# features/extract_patch_pairs.py

import torch
import random
from tqdm import tqdm
from models.backbone import SimSiamBackbone
from utils.image_utils import load_and_preprocess_image
from torchvision import transforms


def extract_patch_pairs(image_dir, save_path, config, num_pos=1000, num_neg=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimSiamBackbone(pretrained=True).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(config['image']['size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['image']['normalization']['mean'],
                             std=config['image']['normalization']['std'])
    ])

    img_list = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    features = []
    for fname in tqdm(img_list, desc="Extracting patches"):
        path = os.path.join(image_dir, fname)
        img = load_and_preprocess_image(path, transform).unsqueeze(0).to(device)
        fmap = model.get_feature_map(img).squeeze(0)  # [C, H, W]
        features.append(fmap.detach().cpu())

    # Create pairs (pos: same image, diff patch / neg: different image)
    pairs = []
    labels = []

    # Positive pairs
    for fmap in features:
        C, H, W = fmap.shape
        for _ in range(num_pos // len(features)):
            y1, x1 = random.randint(0, H-1), random.randint(0, W-1)
            y2, x2 = random.randint(0, H-1), random.randint(0, W-1)
            v1 = fmap[:, y1, x1]
            v2 = fmap[:, y2, x2]
            pairs.append(torch.cat([v1, v2]))
            labels.append(torch.tensor(1.0))

    # Negative pairs
    for _ in range(num_neg):
        fmap1, fmap2 = random.sample(features, 2)
        y1, x1 = random.randint(0, fmap1.shape[1]-1), random.randint(0, fmap1.shape[2]-1)
        y2, x2 = random.randint(0, fmap2.shape[1]-1), random.randint(0, fmap2.shape[2]-1)
        v1 = fmap1[:, y1, x1]
        v2 = fmap2[:, y2, x2]
        pairs.append(torch.cat([v1, v2]))
        labels.append(torch.tensor(0.0))

    X = torch.stack(pairs)  # [N, 2*C]
    y = torch.stack(labels)  # [N]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"pairs": X, "labels": y}, save_path)
    print(f"[✓] Saved patch pairs to {save_path}")


if __name__ == "__main__":
    import yaml
    with open("config.yml", "r") as f:   # 수정된 부분
        config = yaml.safe_load(f)

    extract_patch_pairs(
        image_dir=config['data']['reference_dir'],
        save_path="data/patch_pairs/patch_pairs.pth",
        config=config,
        num_pos=2000,
        num_neg=2000
    )