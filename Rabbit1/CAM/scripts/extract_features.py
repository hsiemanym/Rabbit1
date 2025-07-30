import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# scripts/extract_features.py

import os
import torch
import yaml
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from models.backbone import SimSiamBackbone
from features.embeddings import compute_embedding
from utils.image_utils import load_and_preprocess_image

# ----------------------------
# Load config
# ----------------------------
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# Initialize model
# ----------------------------
model = SimSiamBackbone(proj_dim=256, pretrained=True).to(device)
model.eval()

# ----------------------------
# Image directory
# ----------------------------
image_dir = config['data']['reference_dir']  # e.g., data/rabbits
save_path = config['features']['embedding_cache']  # e.g., features/embeddings.pth

# ----------------------------
# Transformations
# ----------------------------
transform = transforms.Compose([
    transforms.Resize(config['image']['size']),  # e.g., [224, 224]
    transforms.ToTensor(),
    transforms.Normalize(mean=config['image']['normalization']['mean'],
                         std=config['image']['normalization']['std'])
])

# ----------------------------
# Embedding Extraction
# ----------------------------
embeddings = {}

image_files = sorted(os.listdir(image_dir))
for fname in tqdm(image_files, desc="Extracting embeddings"):
    if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    path = os.path.join(image_dir, fname)
    image = load_and_preprocess_image(path, transform).unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        embedding = compute_embedding(model, image)  # [1, 256]
        # ResNet 백본(SimSiamBackbone)에 각 이미지를 통과시켜 임베딩을 얻음
        embeddings[fname] = embedding.squeeze(0).cpu()  # save as [256]

# ----------------------------
# Save
# ----------------------------
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(embeddings, save_path) # 사전에 임베딩 저장
print(f"Saved {len(embeddings)} embeddings to {save_path}")
