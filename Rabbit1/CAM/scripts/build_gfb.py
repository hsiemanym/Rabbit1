import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# scripts/build_gfb.py

import os
import torch
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from tqdm import tqdm

# ---------------------------
# 설정
# ---------------------------
PATCH_FILE = 'features/patches.pth'
SAVE_PATH = 'features/generic_bank.pth'

def option_A(patch_matrix, threshold=0.9, min_count=10):
    """
    Option A: 자주 등장하는 generic patch 식별
    - patch_matrix: [N, D] (전체 이미지에서 수집한 패치 벡터들)
    - threshold: cosine 유사도 임계값
    - min_count: threshold 이상으로 유사한 패치가 등장해야 generic으로 간주
    """
    similarities = cosine_similarity(patch_matrix)
    np.fill_diagonal(similarities, 0.0)

    generic_indices = []
    for i in tqdm(range(similarities.shape[0]), desc="Filtering patches"):
        count = np.sum(similarities[i] >= threshold)
        if count >= min_count:
            generic_indices.append(i)

    generic_vectors = patch_matrix[generic_indices]
    print(f"[Option A] Found {len(generic_vectors)} generic patches")
    return torch.tensor(generic_vectors, dtype=torch.float)


def option_B(patch_matrix, method='kmeans', n_clusters=20):
    """
    Option B: 클러스터링 기반 generic feature 추출
    - patch_matrix: [N, D]
    - method: 'kmeans' or 'dbscan'
    - n_clusters: KMeans 클러스터 수
    """
    if method == 'kmeans':
        km = KMeans(n_clusters=n_clusters, random_state=42)
        km.fit(patch_matrix)
        centers = km.cluster_centers_
        print(f"[Option B] KMeans extracted {len(centers)} cluster centers as generic patches")
        return torch.tensor(centers, dtype=torch.float)

    elif method == 'dbscan':
        db = DBSCAN(eps=0.15, min_samples=10, metric='cosine')
        db.fit(patch_matrix)
        labels = db.labels_
        mask = labels != -1
        generic_vectors = patch_matrix[mask]
        print(f"[Option B] DBSCAN found {generic_vectors.shape[0]} core generic patches")
        return torch.tensor(generic_vectors, dtype=torch.float)

    else:
        raise ValueError("Unknown clustering method")


def main(args):
    assert os.path.exists(PATCH_FILE), f"{PATCH_FILE} not found."
    patch_data = torch.load(PATCH_FILE)  # assumed shape: [N, D] or dict of patches
    if isinstance(patch_data, dict):
        patches = torch.cat(list(patch_data.values()), dim=0)  # [N, D]
    else:
        patches = patch_data  # [N, D]

    patches = patches.cpu()
    patch_matrix = patches.numpy()

    if args.option == 'A':
        gfb = option_A(patch_matrix,
                       threshold=args.threshold,
                       min_count=args.min_count)

    elif args.option == 'B':
        gfb = option_B(patch_matrix,
                       method=args.cluster,
                       n_clusters=args.n_clusters)

    else:
        raise ValueError("option must be 'A' or 'B'")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(gfb, SAVE_PATH)
    print(f"[✓] Saved Generic Feature Bank to: {SAVE_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build Generic Feature Bank (GFB)")
    parser.add_argument('--option', choices=['A', 'B'], default='A', help="Option A (frequency filter) or B (clustering)")
    parser.add_argument('--threshold', type=float, default=0.9, help="Cosine similarity threshold (Option A)")
    parser.add_argument('--min_count', type=int, default=10, help="Minimum occurrence count (Option A)")
    parser.add_argument('--cluster', choices=['kmeans', 'dbscan'], default='kmeans', help="Clustering method (Option B)")
    parser.add_argument('--n_clusters', type=int, default=20, help="Number of clusters (Option B)")

    args = parser.parse_args()
    main(args)
