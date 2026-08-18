
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class HRNetCenterBBox(nn.Module):

    def __init__(self):

        super().__init__()

        # -------------------------------------------------
        # Shared HRNet backbone
        # -------------------------------------------------

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=False,
            num_classes=0
        )

        feature_dim = self.backbone.num_features

        # -------------------------------------------------
        # Center head
        # Same structure as V2
        # -------------------------------------------------

        self.center_head = nn.Sequential(
            nn.Conv2d(
                feature_dim,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                256,
                1,
                kernel_size=1
            )
        )

        # -------------------------------------------------
        # BBox head
        # -------------------------------------------------

        self.bbox_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(
                feature_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                4
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        # Shared features
        features = self.backbone.forward_features(x)

        # -------------------------------------------------
        # Center prediction
        # -------------------------------------------------

        center = self.center_head(
            features
        )

        center = F.interpolate(
            center,
            size=(28, 28),
            mode="bilinear",
            align_corners=False
        )

        # -------------------------------------------------
        # BBox prediction
        # -------------------------------------------------

        pooled = self.bbox_pool(
            features
        )

        pooled = pooled.flatten(
            1
        )

        bbox = self.bbox_head(
            pooled
        )

        return center, bbox
