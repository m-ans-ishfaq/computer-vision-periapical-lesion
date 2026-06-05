# Next Steps (Resume Here)

## Step 0: Wait for download
Check `data/raw/` — the Mendeley dataset (8GB) should contain `Periapical Lesions/` with `Original JPG Images/`, `Augmentation Periapical Lesions/`, and `ImageAnnots/`.

## Step 1: Extract dataset
Extract into `data/raw/` so the structure is:
```
data/raw/Periapical Lesions/
├── Original JPG Images/    *.jpg
├── Augmentation Periapical Lesions/  *.jpg  
└── ImageAnnots/            *.xml
```

## Step 2: Run data preparation (notebooks/01_data_preparation.ipynb)
~2 min. Parses XML, creates train/val/test splits, exports to 3 formats.

## Step 3: Run classification (notebooks/02_classification.ipynb)
~10 min. MobileNetV3 training, outputs `models/classifier_best.pth`.

## Step 4: Run detection (notebooks/03_detection.ipynb)
~15 min. YOLOv8n training, outputs `models/detection_best.pt`.

## Step 5: Run segmentation (notebooks/04_segmentation.ipynb)
~10 min. Mini-UNet training, outputs `models/segmenter_best.pth`.

## Step 6: Launch GUI
```powershell
cd D:\Study\Computer Vision\New\periapical_project
.\venv\Scripts\streamlit run app.py
```

## Step 7: Write paper + record demo (deferred)
---

## Commands to validate project
```powershell
cd D:\Study\Computer Vision\New\periapical_project
.\venv\Scripts\python.exe -c "import sys; sys.path.append('.'); from src.config import *; from src.data_utils import *; from src.models import get_classifier; from src.train_classification import *; from src.train_detection import *; from src.train_segmentation import *; from src.visualize import *; print('ALL IMPORTS OK')"
```
