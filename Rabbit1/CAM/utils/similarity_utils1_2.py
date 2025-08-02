import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# utils/similarity_utils.py

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

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

'''
@torch.no_grad()
def compute_counterfactual_score(fmap_query, fmap_ref, gfb, threshold=0.9):
    """
    - fmap_query: test 이미지의 feature map [C, H, W]
    - fmap_ref: top-1 이미지의 feature map [C, H, W]
    - gfb: [N, C] 형태의 generic feature bank
    - return: [H, W] 형태의 soft mask
    """
    C, H, W = fmap_query.shape
    query_patches = fmap_query.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
    ref_patches = fmap_ref.permute(1, 2, 0).reshape(-1, C)      # [H*W, C]

    query_patches = F.normalize(query_patches, dim=1)
    ref_patches = F.normalize(ref_patches, dim=1)
    gfb = F.normalize(gfb, dim=1)

    scores = []
    for qp in query_patches:
        sim_ref = F.cosine_similarity(qp.unsqueeze(0), ref_patches).max()
        sim_gfb = F.cosine_similarity(qp.unsqueeze(0), gfb).max()

        # Soft: 서로 다를수록 강조
        score = (1 - sim_ref.item()) * (1 - sim_gfb.item())
        scores.append(score)

    return torch.tensor(scores, device=fmap_query.device).view(H, W)
'''


@torch.no_grad()
def compute_counterfactual_score(fmap_query, fmap_ref, gfb, threshold):
    C, H, W = fmap_query.shape
    query_patches = fmap_query.permute(1, 2, 0).reshape(-1, C)
    ref_patches = fmap_ref.permute(1, 2, 0).reshape(-1, C)

    query_patches = F.normalize(query_patches, dim=1)
    ref_patches = F.normalize(ref_patches, dim=1)
    gfb = F.normalize(gfb, dim=1)

    scores = []
    for qp in query_patches:
        ref_sim = F.cosine_similarity(qp.unsqueeze(0), ref_patches).max()
        gfb_sim = F.cosine_similarity(qp.unsqueeze(0), gfb).max()

        # soft counterfactual score (강조)
        cf_score = ((1 - ref_sim) * (1 - gfb_sim)) ** 2
        scores.append(cf_score.item())

    return torch.tensor(scores, device=fmap_query.device).view(H, W)

'''
def generate_patch_based_target(fmap_query, fmap_ref, gfb, threshold):
    """
    fmap_query: [C, H, W]
    fmap_ref: [C, H, W]
    gfb: [N, C]
    return: [H, W] tensor with importance scores
    """
    C, H, W = fmap_query.shape

    # [H*W, C]
    query_patches = fmap_query.permute(1, 2, 0).reshape(-1, C)        # [HW, C]
    ref_patches = fmap_ref.permute(1, 2, 0).reshape(-1, C)            # [HW, C]

    # normalize
    query_patches = F.normalize(query_patches, dim=1)  # [HW, C]
    ref_patches = F.normalize(ref_patches, dim=1)      # [HW, C]
    gfb = F.normalize(gfb, dim=1)                      # [N, C]

    # (1) GFB similarity: [HW, N] → max over N
    sim_to_gfb = F.cosine_similarity(
        query_patches.unsqueeze(1),  # [HW, 1, C]
        gfb.unsqueeze(0),            # [1, N, C]
        dim=-1
    ).max(dim=1).values  # [HW]

    # (2) Ref similarity: [HW, HW] → max over HW
    sim_to_ref = F.cosine_similarity(
        query_patches.unsqueeze(1),  # [HW, 1, C]
        ref_patches.unsqueeze(0),    # [1, HW, C]
        dim=-1
    ).max(dim=1).values  # [HW]

    # (3) Counterfactual score: 두 유사도 모두 낮을 때만 강조
    mask = (sim_to_ref < threshold) & (sim_to_gfb < threshold)  # [HW]
    cf_score = (1 - sim_to_ref) * (1 - sim_to_gfb)              # [HW]
    cf_score = cf_score * mask.float()                          # mask 적용

    return cf_score.view(H, W)  # [H, W]
'''


def generate_patch_based_target(fmap_query, fmap_ref, gfb, threshold, gfb_chunk_size=512, ref_chunk_size=128):
    C, H, W = fmap_query.shape
    HW = H * W

    # [HW, C]
    query_patches = fmap_query.permute(1, 2, 0).reshape(-1, C)
    ref_patches = fmap_ref.permute(1, 2, 0).reshape(-1, C)

    # Normalize
    query_patches = F.normalize(query_patches, dim=1)
    ref_patches = F.normalize(ref_patches, dim=1)
    gfb = F.normalize(gfb, dim=1)

    device = fmap_query.device
    score_map = torch.zeros(HW, device=device)

    for i in range(HW):
        patch = query_patches[i]  # [C]

        # 1. Ref와 최대 유사도
        sim_ref = F.cosine_similarity(patch.unsqueeze(0), ref_patches).max()

        # 2. GFB와 chunked 유사도
        sim_gfb_max = -1.0
        for j in range(0, gfb.size(0), gfb_chunk_size):
            gfb_chunk = gfb[j: j + gfb_chunk_size]  # [chunk, C]
            sim = F.cosine_similarity(patch.unsqueeze(0), gfb_chunk).max()
            sim_gfb_max = max(sim_gfb_max, sim.item())

        # 3. Counterfactual 조건 적용
        if sim_ref < threshold and sim_gfb_max < threshold:
            score_map[i] = (1 - sim_ref) * (1 - sim_gfb_max)

    return score_map.view(H, W)
