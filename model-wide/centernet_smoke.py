import torch
import torch.nn as nn
import torchvision.models as models

class MinimalCenterNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Backbone: Standard ResNet18 
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # CenterNet Heads (basierend auf den 512 Channels vom ResNet18)
        #1. Heatmap: approximation of the probability that a lesion is located at a specific point.
        self.head_heatmap = nn.Conv2d(512, num_classes, kernel_size=1)
        # 2. size : forecast the width and height of the bounding box for each detected lesion.
        self.head_size = nn.Conv2d(512, 2, kernel_size=1)
        # 3. Offset: predict the offset of the center point of the lesion from the nearest pixel location.
        self.head_offset = nn.Conv2d(512, 2, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        heatmap = torch.sigmoid(self.head_heatmap(features))
        size = self.head_size(features)
        offset = self.head_offset(features)
        return heatmap, size, offset

print("Baue minimales CenterNet-Modell...")
model = MinimalCenterNet()
model.eval() # Test-Modus

print("Erzeuge Dummy-Bild (1 Bild, 3 Farbkanäle, 512x512 Pixel)...")
dummy_image = torch.randn(1, 3, 512, 512)

print("Starte Vorhersage (Forward Pass) auf CPU...")
with torch.no_grad():
    out_heatmap, out_size, out_offset = model(dummy_image)

print("\n successfully completed forward pass.")
print(f"Heatmap-Shape (center): {out_heatmap.shape}")
print(f"Size-Shape (width/height): {out_size.shape}")