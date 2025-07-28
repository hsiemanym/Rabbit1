import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# utils/similarity_utils.py

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import torch.nn.functional as F

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
def compute_patchwise_similarity(fmap1, fmap2, model):
    """
    개선된 CCE 방식 patch-wise similarity (max similarity per patch 기준)
    Args:
        fmap1: torch.Tensor [C, H, W] - feature map of query
        fmap2: torch.Tensor [C, H, W] - feature map of reference
        model: PatchSimPredictor
    Returns:
        sim_map: torch.Tensor [H, W] - similarity per patch (from fmap1 to fmap2)
    """
    C, H, W = fmap1.shape
    x1 = fmap1.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
    x2 = fmap2.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]

    x1 = F.normalize(x1, dim=1)  # [N, C]
    x2 = F.normalize(x2, dim=1)  # [N, C]

    sim_scores = []

    for i in range(x1.shape[0]):
        patch1 = x1[i].unsqueeze(0).expand(x2.shape[0], -1)  # [N, C]
        sim_all = model(patch1, x2).squeeze(1)  # [N]
        sim_scores.append(sim_all.max().item())  # 가장 유사한 patch와의 similarity

    sim_map = torch.tensor(sim_scores, dtype=torch.float32).view(H, W).to(fmap1.device)
    return sim_map