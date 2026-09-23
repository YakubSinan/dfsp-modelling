# Skin Lesion Inference Pipeline

An integrated skin lesion inference pipeline combining three trained models:

**WIDE detection → ResNet18 classification → ResNet50 CLOSE localization**

The three models are already integrated into the inference pipeline.

No manual model integration is required. The pipeline automatically connects the WIDE detection model, the ResNet18 lesion classifier, and the ResNet50 CLOSE localization model in the correct order.

This repository contains the trained model weights and the inference code required to run the complete pipeline on an input image.
The models are separate trained components, but they are used together as a single end-to-end inference pipeline.

---

## Pipeline

```text
Input Image
    |
    v
WIDE Model
    |
    | lesion candidate centers
    v
160x160 Crop
    |
    | JPEG round-trip
    v
ResNet18 Classifier
    |
    +---- no_lesion → discarded
    |
    +---- lesion
            |
            v
       ResNet50 CLOSE
            |
            v
     Bounding Box + Center
            |
            v
      Result Image + JSON
```

---

## Models

### 1. WIDE Model

A CenterNet-style model using a ResNet18 backbone.

- Input size: 512 × 512
- Heatmap output: 128 × 128
- Detection threshold: 0.50
- Used to generate candidate lesion centers

Weights:

`models/wide/best_model_aug.pth`

### 2. Lesion Classifier

A ResNet18 image classifier.

Classes:

- `0 = lesion`
- `1 = no_lesion`

- Input size: 224 × 224
- Classification threshold: 0.50

Weights:

`models/classifier/best_lesion_classifier_resnet18.pth`

### 3. CLOSE Localization Model

A ResNet50-based localization model with two output heads:

- Bounding box
- Lesion center

The model predicts normalized coordinates which are mapped back to the original image.

Weights:

`models/close/best_localization_model_v3_center_finetuned.pth`

---

## Project Structure

```text
dfsp-modelling/
|
├── models/
│   ├── wide/
│   │   ├── model.py
│   │   └── best_model_aug.pth
│   │
│   ├── classifier/
│   │   ├── model.py
│   │   └── best_lesion_classifier_resnet18.pth
│   │
│   └── close/
│       ├── model.py
│       └── best_localization_model_v3_center_finetuned.pth
│
├── pipeline/
│   └── inference.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
```

The `.pth` files are stored using Git LFS.

---

## Installation

Git LFS is required because the trained model weights are stored as large files.

Make sure Git LFS is installed and initialized:

```bash
git lfs install

Clone the final integration branch:

git clone -b final-integration https://github.com/YakubSinan/dfsp-modelling.git
cd dfsp-modelling

Install the required Python packages:

pip install -r requirements.txt

## Usage

The complete pipeline can be run with a single command. The user does not need to run the three models separately or manually connect their outputs.

Run the complete inference pipeline with:

```bash
python pipeline/inference.py --image path/to/image.jpg --output-dir outputs
```

The pipeline automatically performs the following steps:

1. Detects candidate lesion centers using the WIDE model.
2. Creates a 160 × 160 crop around each candidate.
3. Classifies each crop using the ResNet18 lesion classifier.
4. Discards candidates classified as no_lesion.
5. Localizes the remaining lesions using the ResNet50 CLOSE model.
6. Maps the predicted bounding boxes and centers back to the original image.
7. Saves the final visualization and JSON results.

Example:

```bash
python pipeline/inference.py --image path/to/image.jpg --output-dir outputs
```

The pipeline can run on CPU and automatically uses CUDA when a compatible GPU is available.

---

## Outputs

For an input such as:

`image.jpg`

the pipeline produces:

```text
outputs/
├── image_result.jpg
└── image_result.json
```

### Result image

The result image contains:

- Predicted lesion bounding boxes
- Predicted lesion centers
- WIDE confidence
- Classifier probability

### Result JSON

The JSON contains:

- Input image name
- Image dimensions
- Number of WIDE candidates
- Number of final lesions
- Bounding boxes
- Lesion centers
- WIDE confidence
- Classifier probability
- Crop information

---

## Pipeline Validation

The integrated pipeline was tested on four images using the final inference implementation.

| Image | WIDE Candidates | Final Lesions |
|---|---:|---:|
| ISIC_3279576.jpg | 1 | 1 |
| ISIC_5288174.jpg | 4 | 4 |
| ISIC_6570415.jpg | 0 | 0 |
| ISIC_9340701.jpg | 5 | 5 |

All four test cases completed successfully.

---

## Notes

This repository is intended for model inference and integration. It does not contain the original model training pipelines or datasets.

The outputs of this pipeline are model predictions and should not be interpreted as a medical diagnosis.

This repository is packaged as a ready-to-run inference system. The trained weights, model definitions, and integration code are included in the repository.
