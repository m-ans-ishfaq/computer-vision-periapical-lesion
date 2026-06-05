import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
DATA_ANNOTATIONS = os.path.join(BASE_DIR, "data", "annotations")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

FULL_TRAIN = False
IMG_SIZE_CLS = 224
IMG_SIZE_DET = 416
IMG_SIZE_SEG = 256
BATCH_SIZE = 8
EPOCHS_CLS = 3 if not FULL_TRAIN else 10
EPOCHS_DET = 3 if not FULL_TRAIN else 20
EPOCHS_SEG = 3 if not FULL_TRAIN else 10
NUM_SAMPLES = 200 if not FULL_TRAIN else None
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
CLASS_NAMES = {3: "PAI_3", 4: "PAI_4", 5: "PAI_5"}
NUM_CLASSES = 3
DEVICE = "cpu"

os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(DATA_ANNOTATIONS, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
