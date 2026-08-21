# ============================================================================
# Folder: model-wide/
# Filename: bridge.py
#
# Translates model peaks into the actual shared/ schema (LesionPrediction,

# ModelOutput) and validates them. Imports shared/ from the project root -

# Does NOT create schema files.

#
# Start: python bridge.py (from the model-wide/ folder)
# ============================================================================
import os
import sys
import inspect
import torch

# Projekt-Root (Elternordner von model-wide/) auf den Pfad, damit shared/ importierbar ist
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shared.schema import LesionPrediction, ModelOutput
from shared import coordinates, validation

from config import OUTPUT_SIZE, BEST_MODEL, DATA_DIR, THRESHOLD, DEVICE
from model import CenterNet
from inference import load_model, extract_peaks


def peaks_to_model_output(peaks, patient_id, image_id):
    """peaks: [(x_grid, y_grid, conf)] im OUTPUT_SIZE-Grid -> ModelOutput."""
    preds = []
    for i, (px, py, conf) in enumerate(peaks):
        x_norm = coordinates.normalize_coordinate(float(px), float(OUTPUT_SIZE))
        y_norm = coordinates.normalize_coordinate(float(py), float(OUTPUT_SIZE))
        preds.append(LesionPrediction(
            patient_id=str(patient_id),
            lesion_id=f"{image_id}_L{i+1}",
            image_id=str(image_id),
            view="wide",
            x=x_norm, y=y_norm,
            confidence=float(conf),
        ))
    return ModelOutput(predictions=preds)


def _verify_shared_source():
    """Druckt Herkunft + Felder von shared/, damit man sieht: es ist das echte."""
    print("shared/schema.py :", inspect.getfile(__import__("shared.schema", fromlist=["x"])))
    fields = getattr(LesionPrediction, "model_fields", None) or getattr(LesionPrediction, "__fields__", {})
    print("Felder           :", list(fields.keys()))


if __name__ == "__main__":
    # Kleiner Selbsttest auf ein paar echten Bildern (Pfade ggf. anpassen)
    _verify_shared_source()
    model = load_model(BEST_MODEL)

    test_images = [
        os.path.join(DATA_DIR, "images", "ISIC_9340701.jpg"),
    ]
    for path in test_images:
        if not os.path.exists(path):
            print("Überspringe (nicht gefunden):", path)
            continue
        import numpy as np
        import torchvision.transforms as T
        from PIL import Image
        from config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD

        tf = T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor(),
                        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        inp = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            hm_prob = torch.sigmoid(model(inp)[0])
        peaks = extract_peaks(hm_prob, threshold=THRESHOLD)

        image_id = os.path.basename(path).split(".")[0]
        patient_id = image_id.split("_")[0] if "_" in image_id else "PATIENT_01"
        out = peaks_to_model_output(peaks, patient_id, image_id)

        validation.validate_model_output(out)   # echten Funktionsnamen ggf. anpassen
        print(f"{image_id}: {len(out.predictions)} Predictions - Validation OK")