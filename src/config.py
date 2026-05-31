import os

# Base directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CUSTOM_TEST_DIR = os.path.join(DATA_DIR, "custom_test")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# Reproducibility
RANDOM_SEED = 42

# Image properties
# Downscale from 300x200 to speed up training while preserving 3:2 aspect ratio.
IMG_HEIGHT = 100
IMG_WIDTH = 150
CHANNELS = 3
NUM_CLASSES = 3

# Data augmentation defaults (train only).
HFLIP_PROB = 0.5
VFLIP_PROB = 0.2
ROTATION_DEGREES = 20
RANDOM_RESIZED_CROP_SCALE = (0.8, 1.0)
RANDOM_RESIZED_CROP_RATIO = (0.9, 1.1)
COLOR_JITTER_BRIGHTNESS = 0.25
COLOR_JITTER_CONTRAST = 0.25
COLOR_JITTER_SATURATION = 0.2

PRINT_INTERVAL = 10