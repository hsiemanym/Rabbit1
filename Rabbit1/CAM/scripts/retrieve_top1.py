import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# scripts/retrieve_top1.py

import random
import numpy as np
import torch
import clip
import yaml
from PIL import Image
import argparse
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from models.backbone import SimSiamBackbone
from features.embeddings import compute_embedding
from utils.image_utils import load_and_preprocess_image




# ------------------------------
# Configuration
# ------------------------------
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

ref_dir = config['paths']['reference_images']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------------
# Initialize model
# ------------------------------
model = SimSiamBackbone(proj_dim=256, pretrained=True).to(device)
model.eval()

# ------------------------------
# Load reference embeddings
# ------------------------------
embedding_dict = torch.load(config['features']['embedding_cache'])  # e.g., features/embeddings.pth
ref_fnames = list(embedding_dict.keys())
ref_embeddings = torch.stack([embedding_dict[f] for f in ref_fnames])  # [N, 256]
ref_embeddings = torch.nn.functional.normalize(ref_embeddings, dim=1)
# ------------------------------
# Image transformation
# ------------------------------
transform = transforms.Compose([
    transforms.Resize(config['image']['size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['image']['normalization']['mean'],
                         std=config['image']['normalization']['std'])
])

def compute_clip_score(image_path1, image_path2):
    """
    CLIPScore를 계산: 두 이미지 경로를 받아서 유사도 출력
    """
    image1 = preprocess(Image.open(image_path1)).unsqueeze(0).to(device)
    image2 = preprocess(Image.open(image_path2)).unsqueeze(0).to(device)

    with torch.no_grad():
        emb1 = clip_model.encode_image(image1).float()
        emb2 = clip_model.encode_image(image2).float()
        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
        score = (emb1 * emb2).sum().item()

    return score

def retrieve_top1(test_img_path):
    # Load and embed test image
    image = load_and_preprocess_image(test_img_path, transform).unsqueeze(0).to(device)  # [1, 3, H, W]
    query = compute_embedding(model, image).cpu().numpy()  # [1, 256]
    query = torch.nn.functional.normalize(query, dim=1).numpy()

    # Compute cosine similarity
    sims = cosine_similarity(query, ref_embeddings.numpy())  # [1, N]
    topk_idx = np.argsort(sims[0])[::-1][:5]
    topk_fnames = [ref_fnames[i] for i in topk_idx]
    topk_scores = sims[0][topk_idx]

    # 예시: 가장 높은 CLIPScore 가진 후보 선택
    best_score = -1
    best_fname = None

    for fname in topk_fnames:
        img_path = os.path.join(config['paths']['reference_images'], fname)
        clip_score = compute_clip_score(test_img_path, img_path)  # 함수 정의 필요
        if clip_score > best_score:
            best_score = clip_score
            best_fname = fname

    print(f"[✓] Test: {test_img_path}")
    print(f"     Top-1 Match: {best_fname}   Similarity: {best_score:.4f}")
    return best_fname, best_score


def main(test_dir):
    test_images = [f for f in os.listdir(test_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for fname in tqdm(test_images, desc="Processing test images"):
        test_path = os.path.join(test_dir, fname)
        retrieve_top1(test_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retrieve Top-1 Similar Image")
    parser.add_argument('--test_dir', type=str, default='data/test/',
                        help="Directory with test images")
    args = parser.parse_args()

    main(args.test_dir)
