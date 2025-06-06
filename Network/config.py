import os

def is_running_on_kaggle():
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

Z_DIM          = 64
C_DIM          = 768
BATCH_SIZE     = 1
EPOCHS         = 3
DEVICE         = "cpu"
IMG_RESOLUTION = 128
CLIP_wWEIGHT   = 0.2
BLUR_FADE_KIMG = 0.1 # after 100 image there will be 0 Blur,

KAGGLE_STR = "/kaggle/working/StyleGAN-T/" if is_running_on_kaggle() else ""
DATASET_PATH   = f"{KAGGLE_STR}/notebooks/Networks/dataset/"