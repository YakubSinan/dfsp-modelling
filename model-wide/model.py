# ============================================================================
# Folder: model-wide/
# Filename: model.py
#
# CenterNet Architecture: ResNet18 Backbone + Deconv Neck + 3 Heads.

# Bias-Init -2.19 on the Heatmap Head (to counter background collapse).

# ============================================================================
import torch.nn as nn
import torchvision.models as models
 
 
class CenterNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # (B,512,16,16)
 
        # Upsampling-Neck: 16 -> 32 -> 64 -> 128  (Stride 32 -> 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
 
        self.hm_head     = nn.Conv2d(64, num_classes, 1)  # Heatmap (Zentren)
        self.wh_head     = nn.Conv2d(64, 2, 1)            # optional (Box-Größe)
        self.offset_head = nn.Conv2d(64, 2, 1)            # optional (Sub-Pixel)
 
        # Bias-Init: Start bei ~sigmoid(-2.19) ≈ 0.1 statt 0.5
        self.hm_head.bias.data.fill_(-2.19)
 
    def forward(self, x):
        feats = self.deconv(self.backbone(x))
        return self.hm_head(feats), self.wh_head(feats), self.offset_head(feats)