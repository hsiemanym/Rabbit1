import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import torch
import torch.nn as nn
import torch.nn.functional as F



class PatchSimPredictor(nn.Module):
        """
        Patch-level similarity predictor (CCE 기반)
        입력: patch1, patch2 (각각 [B, C])
        출력: sim score (B,) - [0, 1] 확률로 해석
        """

        def __init__(self, input_dim=2048, hidden_dim=512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),  # [2048 + 2048 = 4096]
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

        def forward(self, x1, x2):
            pair = torch.cat([x1, x2], dim=1)  # [B, 4096]
            sim = self.net(pair)  # [B, 1]
            return sim

if __name__ == "__main__":
        model = PatchSimPredictor()
        dummy_p1 = torch.randn(8, 2048)
        dummy_p2 = torch.randn(8, 2048)
        out = model(dummy_p1, dummy_p2)
        print(out.shape)  # [8]


if __name__ == "__main__":
    model = PatchSimPredictor()
    dummy_p1 = torch.randn(8, 2048)
    dummy_p2 = torch.randn(8, 2048)
    out = model(dummy_p1, dummy_p2)
    print(out.shape)  # [8]
