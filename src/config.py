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
IMG_HEIGHT = 200
IMG_WIDTH = 300
CHANNELS = 3
NUM_CLASSES = 3

# Default training parameters
DEFAULT_BATCH_SIZE = 32