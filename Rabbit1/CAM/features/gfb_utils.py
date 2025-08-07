# features/gfb_utils.py
import torch
import torch.nn.functional as F


def filter_with_gfb(heatmap, fmap, gfb, threshold=0.7):
    """GFB와 유사도가 낮은 패치 영역만 마스킹하여 히트맵을 필터링합니다."""
    C, H, W = fmap.shape
    mask = torch.zeros((H, W), dtype=torch.float32, device=fmap.device)

    for y in range(H):
        for x in range(W):
            patch = fmap[:, y, x]
            sims = F.cosine_similarity(patch.unsqueeze(0), gfb)
            topk_mean = torch.topk(sims, k=5).values.mean()
            if topk_mean < threshold:
                mask[y, x] = 1.0

    mask_resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=heatmap.shape,
        mode='bilinear',
        align_corners=False
    )

    return heatmap * mask_resized.squeeze().cpu().numpy()