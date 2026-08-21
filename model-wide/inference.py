# ============================================================================
# Folder: model-wide/
# Filename: inference.py
#
# Inference: Load model, Heatmap -> discrete peaks (extract_peaks),

# Overlay visualization. Uses EXACTLY the training normalization.

# ==========================================================================
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

from config import (INPUT_SIZE, OUTPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD,
                    THRESHOLD, DEVICE)
from model import CenterNet

infer_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(ckpt_path):
    """Lädt Checkpoint mit strict=True (bricht laut ab bei Mismatch)."""
    model = CenterNet(num_classes=1, pretrained=False).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def extract_peaks(hm_prob, threshold=THRESHOLD, kernel=3):
    """hm_prob: (1,1,H,W) nach sigmoid. -> [(x,y,conf), ...] im OUTPUT-Grid."""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(hm_prob, kernel, stride=1, padding=pad)
    keep = (hmax == hm_prob).float()
    peaks = (hm_prob * keep)[0, 0]
    ys, xs = torch.where(peaks > threshold)
    confs = peaks[ys, xs]
    return [(int(x), int(y), float(c)) for x, y, c in zip(xs, ys, confs)]


@torch.no_grad()
def infer_and_overlay(model, image_path, threshold=THRESHOLD, save_path=None):
    orig = Image.open(image_path).convert("RGB")
    inp = infer_transform(orig).unsqueeze(0).to(DEVICE)
    hm_prob = torch.sigmoid(model(inp)[0])
    hm_np = hm_prob[0, 0].cpu().numpy()

    peaks = extract_peaks(hm_prob, threshold=threshold)
    scale = INPUT_SIZE / OUTPUT_SIZE
    peaks_512 = [(x * scale, y * scale, c) for (x, y, c) in peaks]

    hm_up = F.interpolate(hm_prob, size=(INPUT_SIZE, INPUT_SIZE),
                          mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    img_512 = np.array(orig.resize((INPUT_SIZE, INPUT_SIZE)))

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(img_512); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(hm_np, cmap="hot", vmin=0, vmax=1)
    ax[1].set_title(f"Heatmap (max {hm_np.max():.3f})"); ax[1].axis("off")
    ax[2].imshow(img_512); ax[2].imshow(hm_up, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    for (x, y, c) in peaks_512:
        ax[2].plot(x, y, "o", ms=14, mec="lime", mfc="none", mew=2.5)
        ax[2].text(x + 8, y, f"{c:.2f}", color="lime", fontsize=11, weight="bold")
    ax[2].set_title(f"Overlay + Peaks ({len(peaks_512)})"); ax[2].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()

    print(f"{image_path}: max {hm_np.max():.4f}, {len(peaks_512)} Peaks (>{threshold})")
    return peaks_512