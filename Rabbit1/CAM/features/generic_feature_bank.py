import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# features/generic_feature_bank.py
'''
features/generic_feature_bank.py (GFB 빌드 로직)
features/gfb_utils.py (GFB 필터링 유틸리티)
scripts/build_gfb.py (GFB 빌드 실행 스크립트)
'''

import torch
import argparse
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from tqdm import tqdm

def build_gfb(patch_file, save_path, option='A', threshold=0.7, min_count=17, method='kmeans', n_clusters=20):
    assert os.path.exists(patch_file), f"{patch_file} not found."
    patch_data = torch.load(patch_file)  # [N, D] or dict

    if isinstance(patch_data, dict):
        patches = torch.cat(list(patch_data.values()), dim=0)
    else:
        patches = patch_data  # [N, D]

    patches = patches.detach().cpu()
    patch_matrix = patches.numpy()

    if option == 'A':
        gfb = _option_A(patch_matrix, threshold, min_count)
    elif option == 'B':
        gfb = _option_B(patch_matrix, method, n_clusters)
    else:
        raise ValueError("GFB option must be 'A' or 'B'.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(gfb, save_path)
    print(f"[✓] GFB saved to {save_path}")
    return gfb


def _option_A(patch_matrix, threshold, min_count):
    """Option A: 자주 등장하는 generic patch 식별"""
    similarities = cosine_similarity(patch_matrix)
    np.fill_diagonal(similarities, 0.0)
    indices = [i for i in range(len(patch_matrix)) if np.sum(similarities[i] >= threshold) >= min_count]
    return torch.tensor(patch_matrix[indices], dtype=torch.float)


def _option_B(patch_matrix, method='kmeans', n_clusters=20):
    """Option B: 클러스터링 기반 generic feature 추출"""
    if method == 'kmeans':
        km = KMeans(n_clusters=n_clusters, random_state=42)
        km.fit(patch_matrix)
        return torch.tensor(km.cluster_centers_, dtype=torch.float)
    elif method == 'dbscan':
        db = DBSCAN(eps=0.15, min_samples=10, metric='cosine')
        db.fit(patch_matrix)
        mask = db.labels_ != -1
        return torch.tensor(patch_matrix[mask], dtype=torch.float)
    else:
        raise ValueError("Unknown clustering method")


# --- GFB 활용 로직 (기존 gfb_utils.py) ---

def filter_with_gfb(heatmap, fmap, gfb, threshold=0.7):
    """GFB와 유사도가 낮은(non-generic) 패치 영역만 마스킹하여 히트맵을 필터링합니다."""
    C, H, W = fmap.shape
    mask = torch.zeros((H, W), dtype=torch.float32, device=fmap.device)

    for y in range(H):
        for x in range(W):
            patch = fmap[:, y, x]
            sims = F.cosine_similarity(patch.unsqueeze(0), gfb)
            if torch.topk(sims, k=5).values.mean() < threshold:
                mask[y, x] = 1.0  # non-generic일 경우 1로 마스킹

    mask_resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=heatmap.shape,
        mode='bilinear',
        align_corners=False
    )
    return heatmap * mask_resized.squeeze().cpu().numpy()

# --- GFB 빌드 실행 스크립트 (기존 scripts/build_gfb.py) ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build Generic Feature Bank (GFB)")
    parser.add_argument('--patch_file', type=str, default='features/patches.pth', help="Path to the patch file.")
    parser.add_argument('--save_path', type=str, default='features/generic_bank.pth', help="Path to save the GFB.")
    parser.add_argument('--option', choices=['A', 'B'], default='A', help="Option A (frequency) or B (clustering).")
    parser.add_argument('--threshold', type=float, default=0.9, help="Similarity threshold for Option A.")
    parser.add_argument('--min_count', type=int, default=10, help="Minimum count for Option A.")
    parser.add_argument('--cluster', choices=['kmeans', 'dbscan'], default='kmeans', help="Clustering method for Option B.")
    parser.add_argument('--n_clusters', type=int, default=20, help="Number of clusters for KMeans.")
    args = parser.parse_args()

    print(f"Building GFB with option '{args.option}'...")
    gfb_tensor = build_gfb(
        patch_file=args.patch_file,
        option=args.option,
        threshold=args.threshold,
        min_count=args.min_count,
        method=args.cluster,
        n_clusters=args.n_clusters
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(gfb_tensor, args.save_path)
    print(f"[✓] Saved Generic Feature Bank to: {args.save_path}")