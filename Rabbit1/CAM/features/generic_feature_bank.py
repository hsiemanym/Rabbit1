import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# features/generic_feature_bank.py

import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN

def build_gfb(patch_file, save_path, option='A', threshold=0.9, min_count=10, method='kmeans', n_clusters=20):
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
    similarities = cosine_similarity(patch_matrix)
    np.fill_diagonal(similarities, 0.0)
    indices = [i for i in range(len(patch_matrix)) if np.sum(similarities[i] >= threshold) >= min_count]
    return torch.tensor(patch_matrix[indices], dtype=torch.float)


def _option_B(patch_matrix, method='kmeans', n_clusters=20):
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
