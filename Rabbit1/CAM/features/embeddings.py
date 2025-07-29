import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn.functional as F

# features/embeddings.py

import torch

def compute_embedding(model, x, no_grad=True):
    """
    Wrapper for getting global image embeddings from SimSiamBackbone.
    Uses model.get_embedding_nograd() or model.get_embedding().
    """
    if no_grad:
        return model.get_embedding_nograd(x)
    else:
        return model.get_embedding(x)
    """
    Compute a global image embedding using SimSiam model
    Args:
        model: SimSiamBackbone
        image_tensor: torch.Tensor of shape [1, 3, H, W]
    Returns:
        torch.Tensor of shape [1, 256]
    """
