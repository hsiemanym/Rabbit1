import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# models/backbone.py

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
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
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # conv5까지
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # GAP

        # Projection head (z)
        self.projection_head = MLPProjectionHead(in_dim=2048, proj_dim=proj_dim)
        self.projector = self.projection_head  # Grad-CAM 호환용 별칭

        # Prediction head (p)
        self.prediction_head = MLPPredictionHead(in_dim=proj_dim)

    def forward(self, x):
        """
        Input: image tensor [B, 3, H, W]
        Output: embedding z, prediction p
        """
        feats = self.encoder(x)              # [B, 2048, 7, 7]
        pooled = self.pool(feats).flatten(1) # [B, 2048]
        z = self.projection_head(pooled)     # [B, proj_dim]
        p = self.prediction_head(z)          # [B, proj_dim]
        return z.detach(), p

    def get_feature_map(self, x):
        """
        Feature map from conv5 layer (before pooling)
        """
        return self.encoder(x)

    def get_embedding(self, x):
        feats = self.encoder(x)
        pooled = self.pool(feats).flatten(1)
        z = self.projection_head(pooled)
        return z

    def get_embedding_nograd(self, x):
        with torch.no_grad():
            return self.get_embedding(x)

    def forward_backbone(self, x):
        """
        Just forward encoder
        """
        return self.encoder(x)

    def eval(self):
        super().eval()
        self.encoder.eval()
        self.pool.eval()
        self.projection_head.eval()
        self.prediction_head.eval()
        return self