
import argparse
import json
import os
import sys

# ============================================================
# REPOSITORY ROOT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from models.wide.model import CenterNet
from models.classifier.model import create_model
from models.close.model import LocalizationModel


# ============================================================
# CONFIG
# ============================================================

WIDE_INPUT_SIZE = 512
WIDE_THRESHOLD = 0.50

CROP_SIZE = 160
JPEG_QUALITY = 95

CLASSIFIER_THRESHOLD = 0.50
CLASSIFIER_IMAGE_SIZE = 224

CLOSE_IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


WIDE_WEIGHTS = os.path.join(
    BASE_DIR,
    "models",
    "wide",
    "best_model_aug.pth"
)

CLASSIFIER_WEIGHTS = os.path.join(
    BASE_DIR,
    "models",
    "classifier",
    "best_lesion_classifier_resnet18.pth"
)

CLOSE_WEIGHTS = os.path.join(
    BASE_DIR,
    "models",
    "close",
    "best_localization_model_v3_center_finetuned.pth"
)


# ============================================================
# TRANSFORMS
# ============================================================

wide_transform = transforms.Compose([
    transforms.Resize(
        (WIDE_INPUT_SIZE, WIDE_INPUT_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    ),
])


classifier_transform = transforms.Compose([
    transforms.Resize(
        (CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    ),
])


close_transform = transforms.Compose([
    transforms.Resize(
        (CLOSE_IMAGE_SIZE, CLOSE_IMAGE_SIZE)
    ),
    transforms.ToTensor(),
])


# ============================================================
# LOAD MODELS
# ============================================================

def load_models(device):

    # -------------------------
    # WIDE
    # -------------------------

    wide_state = torch.load(
        WIDE_WEIGHTS,
        map_location=device,
        weights_only=False
    )

    wide_model = CenterNet(
        num_classes=1,
        pretrained=False
    )

    wide_model.load_state_dict(
        wide_state
    )

    wide_model.to(device)
    wide_model.eval()


    # -------------------------
    # CLASSIFIER
    # -------------------------

    classifier_checkpoint = torch.load(
        CLASSIFIER_WEIGHTS,
        map_location=device,
        weights_only=False
    )

    classifier_model = create_model(
        num_classes=2
    )

    classifier_model.load_state_dict(
        classifier_checkpoint[
            "model_state_dict"
        ]
    )

    classifier_model.to(device)
    classifier_model.eval()


    # -------------------------
    # CLOSE
    # -------------------------

    close_checkpoint = torch.load(
        CLOSE_WEIGHTS,
        map_location=device,
        weights_only=False
    )

    close_model = LocalizationModel(
        pretrained=False
    )

    close_model.load_state_dict(
        close_checkpoint[
            "model_state_dict"
        ]
    )

    close_model.to(device)
    close_model.eval()


    return (
        wide_model,
        classifier_model,
        close_model
    )


# ============================================================
# WIDE PEAK EXTRACTION
# ============================================================

def extract_peaks(
    hm_prob,
    threshold=WIDE_THRESHOLD,
    kernel=3
):

    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(
        hm_prob,
        kernel,
        stride=1,
        padding=pad
    )

    keep = (
        hmax == hm_prob
    ).float()

    peaks = (
        hm_prob * keep
    )[0, 0]

    ys, xs = torch.where(
        peaks > threshold
    )

    confs = peaks[
        ys,
        xs
    ]

    return [
        (
            int(x),
            int(y),
            float(c)
        )
        for x, y, c
        in zip(
            xs,
            ys,
            confs
        )
    ]


# ============================================================
# SAFE CROP
# ============================================================

def crop_around_point(
    image,
    center_x,
    center_y,
    crop_size=CROP_SIZE
):

    h, w = image.shape[:2]

    half = crop_size // 2

    x1 = max(
        0,
        center_x - half
    )

    y1 = max(
        0,
        center_y - half
    )

    x2 = min(
        w,
        x1 + crop_size
    )

    y2 = min(
        h,
        y1 + crop_size
    )

    x1 = max(
        0,
        x2 - crop_size
    )

    y1 = max(
        0,
        y2 - crop_size
    )

    crop = image[
        y1:y2,
        x1:x2
    ]

    return (
        crop,
        x1,
        y1
    )


# ============================================================
# JPEG ROUND TRIP
# ============================================================

def jpeg_roundtrip(
    crop,
    quality=JPEG_QUALITY
):

    success, encoded = cv2.imencode(
        ".jpg",
        crop,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality
        ]
    )

    if not success:
        raise RuntimeError(
            "JPEG encoding failed."
        )

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )

    if decoded is None:
        raise RuntimeError(
            "JPEG decoding failed."
        )

    return decoded


# ============================================================
# CLASSIFIER
# ============================================================

@torch.no_grad()
def classify_crop(
    classifier_model,
    crop,
    device
):

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        crop_rgb
    )

    tensor = classifier_transform(
        pil_image
    )

    tensor = tensor.unsqueeze(
        0
    ).to(device)

    logits = classifier_model(
        tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    lesion_probability = float(
        probabilities[
            0,
            0
        ].item()
    )

    prediction = (
        "lesion"
        if lesion_probability >= CLASSIFIER_THRESHOLD
        else "no_lesion"
    )

    return (
        prediction,
        lesion_probability
    )


# ============================================================
# CLOSE LOCALIZATION
# ============================================================

@torch.no_grad()
def localize_crop(
    close_model,
    crop,
    device
):

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        crop_rgb
    )

    tensor = close_transform(
        pil_image
    )

    tensor = tensor.unsqueeze(
        0
    ).to(device)

    bbox, center = close_model(
        tensor
    )

    bbox = bbox[
        0
    ].cpu().numpy()

    center = center[
        0
    ].cpu().numpy()

    return (
        bbox,
        center
    )


# ============================================================
# MAP TO ORIGINAL IMAGE
# ============================================================

def map_to_original(
    bbox,
    center,
    crop_x,
    crop_y,
    crop_width,
    crop_height
):

    cx_norm, cy_norm = center

    cx_crop = (
        cx_norm *
        crop_width
    )

    cy_crop = (
        cy_norm *
        crop_height
    )

    center_original = [
        float(
            crop_x + cx_crop
        ),
        float(
            crop_y + cy_crop
        )
    ]


    bbox_cx_norm = bbox[0]
    bbox_cy_norm = bbox[1]
    bbox_w_norm = bbox[2]
    bbox_h_norm = bbox[3]


    bbox_cx = (
        bbox_cx_norm *
        crop_width
    )

    bbox_cy = (
        bbox_cy_norm *
        crop_height
    )

    bbox_w = (
        bbox_w_norm *
        crop_width
    )

    bbox_h = (
        bbox_h_norm *
        crop_height
    )


    x1 = (
        crop_x +
        bbox_cx -
        bbox_w / 2
    )

    y1 = (
        crop_y +
        bbox_cy -
        bbox_h / 2
    )

    x2 = (
        crop_x +
        bbox_cx +
        bbox_w / 2
    )

    y2 = (
        crop_y +
        bbox_cy +
        bbox_h / 2
    )


    bbox_original = [
        float(x1),
        float(y1),
        float(x2),
        float(y2)
    ]

    return (
        bbox_original,
        center_original
    )


# ============================================================
# DRAW RESULTS
# ============================================================

def draw_results(
    image,
    detections
):

    output = image.copy()

    for detection in detections:

        bbox = detection[
            "bbox"
        ]

        center = detection[
            "center"
        ]


        x1, y1, x2, y2 = [
            int(round(v))
            for v in bbox
        ]

        cx, cy = [
            int(round(v))
            for v in center
        ]


        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            output,
            (cx, cy),
            4,
            (0, 0, 255),
            -1
        )


        label = (
            f"WIDE "
            f"{detection['wide_confidence']:.2f} "
            f"| CLS "
            f"{detection['classifier_probability']:.2f}"
        )


        cv2.putText(
            output,
            label,
            (
                x1,
                max(
                    20,
                    y1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return output


# ============================================================
# FULL PIPELINE
# ============================================================

@torch.no_grad()
def run_pipeline(
    image_path,
    output_dir="outputs"
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    original = image.copy()

    original_height, original_width = (
        image.shape[:2]
    )


    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    (
        wide_model,
        classifier_model,
        close_model
    ) = load_models(
        device
    )


    # --------------------------------------------------------
    # WIDE
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    wide_input = wide_transform(
        Image.fromarray(
            image_rgb
        )
    )

    wide_input = wide_input.unsqueeze(
        0
    ).to(device)


    wide_output = wide_model(
        wide_input
    )

    heatmap_logits = wide_output[0]

    heatmap_probability = torch.sigmoid(
        heatmap_logits
    )


    peaks = extract_peaks(
        heatmap_probability,
        threshold=WIDE_THRESHOLD
    )

    print(
        f"WIDE candidates: {len(peaks)}"
    )


    # --------------------------------------------------------
    # WIDE -> ORIGINAL COORDINATES
    # --------------------------------------------------------

    scale_x = (
        original_width /
        WIDE_INPUT_SIZE
    )

    scale_y = (
        original_height /
        WIDE_INPUT_SIZE
    )


    detections = []


    # --------------------------------------------------------
    # CANDIDATES
    # --------------------------------------------------------

    for peak_index, (
        peak_x,
        peak_y,
        wide_confidence
    ) in enumerate(peaks):


        # Heatmap 128x128 -> 512x512
        x_512 = (
            peak_x * 4
        )

        y_512 = (
            peak_y * 4
        )


        x_original = int(
            round(
                x_512 * scale_x
            )
        )

        y_original = int(
            round(
                y_512 * scale_y
            )
        )


        # ----------------------------------------------------
        # CROP
        # ----------------------------------------------------

        crop, crop_x, crop_y = (
            crop_around_point(
                original,
                x_original,
                y_original
            )
        )


        if crop.size == 0:
            continue


        crop = jpeg_roundtrip(
            crop,
            JPEG_QUALITY
        )


        crop_height, crop_width = (
            crop.shape[:2]
        )


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        prediction, lesion_probability = (
            classify_crop(
                classifier_model,
                crop,
                device
            )
        )


        print(
            f"Candidate {peak_index + 1}: "
            f"WIDE={wide_confidence:.4f}, "
            f"classifier={lesion_probability:.4f}, "
            f"prediction={prediction}"
        )


        if prediction != "lesion":
            continue


        # ----------------------------------------------------
        # CLOSE
        # ----------------------------------------------------

        bbox, center = (
            localize_crop(
                close_model,
                crop,
                device
            )
        )


        bbox_original, center_original = (
            map_to_original(
                bbox,
                center,
                crop_x,
                crop_y,
                crop_width,
                crop_height
            )
        )


        detections.append({

            "wide_confidence":
                float(
                    wide_confidence
                ),

            "classifier_probability":
                float(
                    lesion_probability
                ),

            "bbox":
                bbox_original,

            "center":
                center_original,

            "crop_origin": [
                int(crop_x),
                int(crop_y)
            ],

            "crop_size": [
                int(crop_width),
                int(crop_height)
            ]
        })


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    base_name = os.path.splitext(
        os.path.basename(
            image_path
        )
    )[0]


    output_image_path = os.path.join(
        output_dir,
        f"{base_name}_result.jpg"
    )

    output_json_path = os.path.join(
        output_dir,
        f"{base_name}_result.json"
    )


    result_image = draw_results(
        original,
        detections
    )


    cv2.imwrite(
        output_image_path,
        result_image
    )


    result = {

        "image":
            os.path.basename(
                image_path
            ),

        "image_size": [
            int(original_width),
            int(original_height)
        ],

        "wide_threshold":
            WIDE_THRESHOLD,

        "classifier_threshold":
            CLASSIFIER_THRESHOLD,

        "num_wide_candidates":
            len(peaks),

        "num_lesions":
            len(detections),

        "detections":
            detections
    }


    with open(
        output_json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )


    print()
    print(
        "=" * 60
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 60
    )

    print(
        f"WIDE candidates : "
        f"{len(peaks)}"
    )

    print(
        f"Lesions         : "
        f"{len(detections)}"
    )

    print(
        f"Image output    : "
        f"{output_image_path}"
    )

    print(
        f"JSON output     : "
        f"{output_json_path}"
    )


    return result


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Skin lesion inference pipeline: "
            "WIDE -> Classifier -> CLOSE"
        )
    )


    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )


    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for outputs"
    )


    args = parser.parse_args()


    run_pipeline(
        image_path=args.image,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
