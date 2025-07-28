import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# scripts/retrieve_top1.py

import os
import torch
import yaml
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ------------------------------
# Image transformation
# ------------------------------
transform = transforms.Compose([
    transforms.Resize(config['image']['size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['image']['normalization']['mean'],
                         std=config['image']['normalization']['std'])
])

def retrieve_top1(test_img_path):
    # Load and embed test image
    image = load_and_preprocess_image(test_img_path, transform).unsqueeze(0).to(device)  # [1, 3, H, W]
    query = compute_embedding(model, image).cpu().numpy()  # [1, 256]

    # Compute cosine similarity
    sims = cosine_similarity(query, ref_embeddings.numpy())  # [1, N]
    top_idx = sims.argmax()
    top_score = sims[0, top_idx]
    top_match = ref_fnames[top_idx]

    print(f"[✓] Test: {test_img_path}")
    print(f"     Top-1 Match: {top_match}   Similarity: {top_score:.4f}")
    return top_match, top_score


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
