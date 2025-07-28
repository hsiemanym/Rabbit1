import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# features/embeddings.py

import torch

def compute_embedding(model, image_tensor):
    """
    Compute a global image embedding using SimSiam model
    Args:
        model: SimSiamBackbone
        image_tensor: torch.Tensor of shape [1, 3, H, W]
    Returns:
        torch.Tensor of shape [1, 256]
    """
    model.eval()
    with torch.set_grad_enabled(True):  # ← gradient 추적 가능하게 변경
        embedding = model.get_embedding(image_tensor)  # [1, D]
    return embedding