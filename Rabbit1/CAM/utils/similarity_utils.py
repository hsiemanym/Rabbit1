import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# utils/similarity_utils.py

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

def cosine_similarity_torch(x1: torch.Tensor, x2: torch.Tensor) -> float:
    """
    PyTorch 기반 cosine similarity 계산
    Args:
        x1, x2: torch.Tensor of shape [D]
    Returns:
        float in [0, 1]
    """
    return torch.nn.functional.cosine_similarity(x1.unsqueeze(0), x2.unsqueeze(0), dim=1).item()

def cosine_similarity_np(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    NumPy 기반 cosine similarity 계산
    Args:
        x1, x2: numpy arrays of shape [D]
    Returns:
        float in [0, 1]
    """
    x1_norm = x1 / (np.linalg.norm(x1) + 1e-8)
    x2_norm = x2 / (np.linalg.norm(x2) + 1e-8)
    return float(np.dot(x1_norm, x2_norm))

def pairwise_cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    전체 벡터 간 pairwise cosine similarity matrix
    Args:
        vectors: numpy array of shape [N, D]
    Returns:
        [N, N] cosine similarity matrix
    """
    return sklearn_cosine(vectors)

def euclidean_distance(x1, x2):
    """
    유클리드 거리 계산 (NumPy 또는 Torch 모두 지원)
    """
    if isinstance(x1, torch.Tensor):
        return torch.norm(x1 - x2).item()
    elif isinstance(x1, np.ndarray):
        return float(np.linalg.norm(x1 - x2))
    else:
        raise TypeError("Inputs must be torch.Tensor or np.ndarray")

def rank_similarities(query, candidates, topk=1, use_torch=True):
    """
    유사도 순으로 후보 정렬 후 상위 top-k 반환
    Args:
        query: [D]
        candidates: [N, D]
        topk: int
        use_torch: whether to use torch or numpy
    Returns:
        list of (index, score)
    """
    scores = []
    for i, c in enumerate(candidates):
        sim = cosine_similarity_torch(query, c) if use_torch else cosine_similarity_np(query, c)
        scores.append((i, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topk]

@torch.no_grad()
def compute_counterfactual_score(fmap_query, fmap_ref, gfb, threshold=0.9):
    """
    각 query patch가 reference와 GFB 어디에도 없을 경우 counterfactual로 간주하여 강조
    Returns:
        counterfactual mask: [H, W]
    """
    C, H, W = fmap_query.shape
    query_patches = fmap_query.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
    ref_patches = fmap_ref.permute(1, 2, 0).reshape(-1, C)      # [H*W, C]

    query_patches = F.normalize(query_patches, dim=1)
    ref_patches = F.normalize(ref_patches, dim=1)
    gfb = F.normalize(gfb, dim=1)

    mask = []
    for qp in query_patches:
        # Reference와 최대 유사도
        ref_sim = torch.nn.functional.cosine_similarity(qp.unsqueeze(0), ref_patches).max()
        gfb_sim = torch.nn.functional.cosine_similarity(qp.unsqueeze(0), gfb).max()

        if ref_sim < threshold and gfb_sim < threshold:
            mask.append(1.0)  # counterfactual
        else:
            mask.append(0.0)
    return torch.tensor(mask, device=fmap_query.device).view(H, W)

