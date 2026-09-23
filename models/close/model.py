import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class LocalizationModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None

        backbone = resnet50(weights=weights)

        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )

        self.center_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)

        bbox = self.bbox_head(features)
        center = self.center_head(features)

        return bbox, center