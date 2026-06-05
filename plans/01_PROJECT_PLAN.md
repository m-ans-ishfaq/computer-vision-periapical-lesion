# Periapical Lesion Classification, Detection & Segmentation

## Overview
End-to-end computer vision project for dental periapical lesion analysis using a public dataset from Data in Brief (2024). Three tasks: classification (PAI severity grade), detection (bounding boxes), and weakly-supervised segmentation (boxes-to-masks).

## Dataset
- **Source:** "A Dataset of apical periodontitis lesions in panoramic radiographs" (Data in Brief, Vol 54, 2024, S2352340924004554)
- **Mendeley DOI:** 10.17632/kx52tk2ddj.3
- **Download URL:** https://data.mendeley.com/datasets/kx52tk2ddj/3
- **Expected extraction location:** `data/raw/`
- **Structure after extraction:**
  ```
  data/raw/Periapical Lesions/
  ├── Original JPG Images/    # 3,926 original images
  ├── Augmentation Periapical Lesions/  # 17,004 augmented images
  └── ImageAnnots/            # XML LabelMe annotations
  ```
- **Classes:** PAI 3 (mild), PAI 4 (moderate), PAI 5 (severe)
- **Labels:** `PAI 3` → 3, `PAI 4` → 4, `PAI 5` → 5
- **Annotations:** Bounding boxes only (no pixel masks)

## Key Constraints
- **CPU-only training** (Intel UHD Graphics 620, no NVIDIA GPU)
- **Python 3.13.13**, PyTorch 2.12.0+cpu
- **Small subset (200 images)** for demo; FULL_TRAIN flag for overnight full training
- **All deliverables as .ipynb notebooks**
- **Streamlit GUI** for visualization

## Project Structure
```
periapical_project/
├── app.py                   # Streamlit GUI (4 tabs)
├── requirements.txt         # Dependencies
├── generate_notebooks.py    # Notebook generator
├── src/
│   ├── config.py            # Paths, hyperparameters, magic numbers
│   ├── data_utils.py        # XML parsing, dataset scanning, format conversion
│   ├── models.py            # MobileNetV3 classifier, UNetMini segmenter
│   ├── train_classification.py  # Training loop + evaluation
│   ├── train_detection.py       # YOLOv8 training wrapper
│   ├── train_segmentation.py    # UNet training + Dice/IoU
│   └── visualize.py         # Plots, confusion matrix, overlays
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_classification.ipynb
│   ├── 03_detection.ipynb
│   └── 04_segmentation.ipynb
├── plans/                   # This directory
├── data/ (raw/, processed/, annotations/)
├── models/ (saved checkpoints)
└── outputs/ (metrics, plots)
```

## Models
| Task | Model | Input Size | Output | Params |
|------|-------|-----------|--------|--------|
| Classification | MobileNetV3-Small (pretrained) | 224x224 | 3 classes | 1.1M |
| Detection | YOLOv8n (pretrained) | 640x640 | boxes + class | 3.2M |
| Segmentation | Mini-UNet (scratch) | 256x256 | 4-channel mask (bg + 3 classes) | 1.9M |

## Execution Order (Run When Dataset is Ready)

### Step 1: Extract dataset
Extract the Mendeley download into `data/raw/` so `data/raw/Periapical Lesions/` exists.

### Step 2: `01_data_preparation.ipynb` (~2 min)
- Scans `data/raw/` for images + XML annotations
- Builds manifest with all metadata
- Splits into train/val/test (70/15/15)
- Exports:
  - Classification: `data/processed/classification/` (labeled subdirs with symlinks/copies)
  - Detection: `data/processed/detection/` (images + YOLO-format labels + dataset.yaml)
  - Segmentation: `data/processed/segmentation/` (images + PNG masks)

### Step 3: `02_classification.ipynb` (~10 min)
- Trains MobileNetV3-Small on 224x224 images
- 3 epochs (demo) / 20 epochs (full)
- Outputs: `models/classifier_best.pth`, confusion matrix + ROC curves in `outputs/`

### Step 4: `03_detection.ipynb` (~15 min)
- Trains YOLOv8n on 640x640 images
- 3 epochs (demo) / 50 epochs (full)
- Outputs: `models/detection_best.pt`, mAP plot + PR curves in `outputs/`

### Step 5: `04_segmentation.ipynb` (~10 min)
- Trains Mini-UNet on 256x256 images with pseudo-masks from boxes
- 3 epochs (demo) / 30 epochs (full)
- Outputs: `models/segmenter_best.pth`, Dice/IoU over epochs + overlay examples in `outputs/`

### Step 6: Streamlit GUI
```bash
streamlit run app.py
```
Tabs: Dashboard (summary metrics), Classification (upload + predict), Detection (upload + detect), Segmentation (upload + segment)

## Research Angle
**Weakly-supervised semantic segmentation from bounding box annotations.**
- Dataset has only bounding boxes (no pixel masks)
- We generate binary masks from box regions (OpenCV `rectangle fill`)
- This is the key novelty claim: demonstrating that segmentation can work as a downstream task from box-only annotations
- If results are decent, strengthens the paper

## Critical Design Decisions
1. **Label remapping:** XML uses PAI numbers (3,4,5); internally we remap: 3→1, 4→2, 5→3 for masks, or subtract 3 for detection/classification indices
2. **Seg mask values:** 0=background, 1=PAI_3, 2=PAI_4, 3=PAI_5 → matches CrossEntropyLoss with 4 output channels
3. **Notebook path resolution:** Tries CWD then parent, looking for `src/config.py` as anchor
4. **FULL_TRAIN flag:** Config toggle in `config.py`; when False, uses 200 samples, small img sizes, 3 epochs
5. **Streamlit threading:** Single-file app, no multiprocessing needed
6. **Augmented images ignored by default:** Only 3,926 originals used (can be enabled in config)

## Known Issues / Gotchas
- **Masks at borders:** Nearest-neighbor resize may introduce artifacts at object boundaries (acceptable for demo)
- **YOLO on CPU:** Very slow for full training; demo mode with 3 epochs on 200 images takes ~15 min
- **Memory:** Batch size 8 on CPU for 256x256 segmentation should fit 8GB RAM; reduce to 4 if OOM
- **Empty images:** Some images in the dataset may have no annotations; they are filtered out with a warning
- **scan_dataset():** Walks `data/raw/` looking for directories with "original"+"image", "augment", or "annot" in name

## Post-Completion (Deferred)
1. Write research paper (~4-6 pages, template: Elsevier or IEEE)
2. Record video demo (3-5 min screen capture with narration)
