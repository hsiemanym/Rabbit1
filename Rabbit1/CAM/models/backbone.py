import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# models/backbone.py

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class MLPProjectionHead(nn.Module):
    """
    SimSiam-style 3-layer projection head
    (e.g., 2048 -> 2048 -> 2048 -> 256)
    """
    def __init__(self, in_dim=2048, proj_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, proj_dim),
            nn.BatchNorm1d(proj_dim, affine=False)  # no bias/scale
        )

    def forward(self, x):
        return self.net(x)


class MLPPredictionHead(nn.Module):
    """
    SimSiam-style prediction head (2-layer MLP)
    (e.g., 256 -> 512 -> 256)
    """
    def __init__(self, in_dim=256, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        return self.net(x)


class SimSiamBackbone(nn.Module):
    """
    SimSiam backbone with ResNet-50 encoder and MLP heads
    """
    def __init__(self, proj_dim=256, pretrained=True):
        super().__init__()

        # Base encoder: ResNet-50 without FC

        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # up to conv5
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # GAP to get embedding

        # Flattened embedding dimension (ResNet50 conv5 output = 2048)
        self.projection_head = MLPProjectionHead(in_dim=2048, proj_dim=proj_dim)
        self.projector = self.projection_head  # alias for Grad-CAM compatibility
        self.prediction_head = MLPPredictionHead(in_dim=proj_dim)

    def forward(self, x):
        """
        Input: image tensor [B, 3, H, W]
        Output: embedding z, projection p
        """
        feats = self.encoder(x)              # [B, 2048, 7, 7]
        pooled = self.pool(feats).flatten(1) # [B, 2048]
        z = self.projection_head(pooled)     # [B, proj_dim]
        p = self.prediction_head(z)          # [B, proj_dim]
        return z.detach(), p  # z: stop-gradient branch, p: prediction

    def get_feature_map(self, x):
        """
        Returns spatial feature map from conv5 layer (before pooling)
        Used for patch-level Grad-CAM or GFB
        """
        return self.encoder(x)  # [B, 2048, 7, 7]

    def get_embedding(self, x):
        """
        Returns projection vector only (z) after GAP and projection head
        """
        with torch.no_grad():
            feats = self.encoder(x)
            pooled = self.pool(feats).flatten(1)
            z = self.projection_head(pooled)
        return z

    def forward_backbone(self, x):
        return self.encoder(x)