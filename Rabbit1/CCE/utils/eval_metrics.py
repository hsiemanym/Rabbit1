import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# utils/eval_metrics.py

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_auc_score

def pixelwise_correlation(heatmap1, heatmap2):
    """
    heatmap1, heatmap2: numpy arrays of shape [H, W], normalized [0, 1]
    Returns: float correlation in [-1, 1]
    """
    h1 = heatmap1.flatten()
    h2 = heatmap2.flatten()
    if np.std(h1) == 0 or np.std(h2) == 0:
        return 0.0
    return np.corrcoef(h1, h2)[0, 1]

def ssim_heatmap_similarity(heatmap1, heatmap2):
    """
    Structural Similarity Index between two heatmaps
    Returns: float in [0, 1]
    """
    return ssim(heatmap1, heatmap2, data_range=1.0)

def auc_score_vs_binary_mask(heatmap, mask, threshold=0.5):
    """
    AUC score measuring alignment between continuous heatmap and binary ground-truth mask
    Args:
        heatmap: numpy array [H, W], float [0, 1]
        mask: numpy array [H, W], binary {0, 1}
    """
    h_flat = heatmap.flatten()
    m_flat = mask.flatten()
    if len(np.unique(m_flat)) < 2:
        return 0.0  # invalid mask
    return roc_auc_score(m_flat, h_flat)

def heatmap_confidence_score(heatmap):
    """
    Computes the concentration/confidence of a heatmap:
    e.g., how peaked vs. dispersed it is
    Returns: entropy-like score (lower = more confident)
    """
    heatmap = heatmap.flatten()
    heatmap /= (np.sum(heatmap) + 1e-8)
    return -np.sum(heatmap * np.log(heatmap + 1e-8))

def overlap_ratio(heatmap, mask, threshold=0.5):
    """
    IOU-style overlap ratio between thresholded heatmap and binary mask
    """
    bin_heatmap = (heatmap >= threshold).astype(np.uint8)
    intersection = np.logical_and(bin_heatmap, mask).sum()
    union = np.logical_or(bin_heatmap, mask).sum()
    return intersection / (union + 1e-8)
