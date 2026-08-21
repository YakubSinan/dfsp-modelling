# central configuration - 
#  changing Colab -> lokal -> HPC only this data 
# ============================================================================
import torch
 
# ---  / Grid ---
INPUT_SIZE = 512
OUTPUT_STRIDE = 4
OUTPUT_SIZE = INPUT_SIZE // OUTPUT_STRIDE   # 128
 
# ---normalization (ImageNet, wegen pretrained Backbone) ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
 
# --- Pfade (JE UMGEBUNG ANPASSEN) ---
DATA_DIR   = "/content/itobos_data"
CKPT_DIR   = "/content/drive/MyDrive/dfsp_wide_model/checkpoints"
BEST_MODEL = f"{CKPT_DIR}/best_model_aug.pth"
 
# --- Training ---
BATCH_SIZE = 16
LR         = 1.25e-4
EPOCHS     = 20
PATIENCE   = 5
TRAIN_SIZE = 8473
SEED       = 42
 
# --- Inferenz ---
THRESHOLD = 0.3
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")