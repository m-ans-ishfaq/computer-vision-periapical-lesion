import json
import os

NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def make_cell(cell_type, source_lines):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source_lines],
    }

def make_code(source_lines):
    c = make_cell("code", source_lines)
    c["outputs"] = []
    return c

def make_md(source_lines):
    return make_cell("markdown", source_lines)

KERNEL_SPEC = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}

LANG_INFO = {
    "name": "python",
    "version": "3.13.13",
}

def write_notebook(filename, cells):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": KERNEL_SPEC,
            "language_info": LANG_INFO,
        },
        "cells": cells,
    }
    path = os.path.join(NOTEBOOKS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Created: {path}")

def notebook_01():
    cells = [
        make_md([
            "# 01 - Data Preparation",
            "Parsing the periapical lesion dataset XML annotations, building manifests, and creating train/val/test splits for classification, detection, and segmentation.",
        ]),
        make_code([
            "import sys, os",
            "# Find project root (works from project root or notebooks/ subdir)",
            "_root = None",
            "for _p in [os.path.abspath('.'), os.path.abspath('..')]:",
            "    if os.path.exists(os.path.join(_p, 'src', 'config.py')):",
            "        _root = _p",
            "        break",
            "if _root:",
            "    sys.path.append(_root)",
            "else:",
            "    raise RuntimeError('Could not find project root (src/config.py not found)')",
            "from src.data_utils import *",
            "from src.config import *",
            "import random",
            "random.seed(RANDOM_SEED)",
        ]),
        make_code([
            "# Scan dataset directories",
            "original_dir, aug_dir, annot_dir = scan_dataset()",
            "if original_dir is None or annot_dir is None:",
            "    raise FileNotFoundError(f'Could not find dataset in {DATA_RAW}. '\n"
            "                              'Extract the Mendeley download into data/raw/.')",
            "print(f'Original images dir: {original_dir}')",
            "print(f'Annotation dir: {annot_dir}')",
        ]),
        make_code([
            "# Build manifest from original images only (not augmented)",
            "manifest = build_dataset_manifest(original_dir, annot_dir, sample_size=NUM_SAMPLES)",
            "print(f'Total samples in manifest: {len(manifest)}')",
        ]),
        make_code([
            "# Check class distribution",
            "dist = get_class_distribution(manifest)",
            "print('Class distribution:', dist)",
        ]),
        make_code([
            "# Split into train/val/test",
            "splits = split_dataset(manifest)",
            "for k, v in splits.items():",
            "    print(f'{k}: {len(v)} samples')",
        ]),
        make_md([
            "## Prepare Classification Data",
        ]),
        make_code([
            "cls_records = prepare_classification_data(splits)",
            "print(f'Classification records: {len(cls_records)}')",
        ]),
        make_md([
            "## Prepare Detection Data (YOLO format)",
        ]),
        make_code([
            "det_records, yaml_path = prepare_detection_data(splits)",
            "print(f'Detection records: {len(det_records)}')",
            "print(f'YAML config: {yaml_path}')",
        ]),
        make_md([
            "## Prepare Segmentation Data (Box-to-Mask conversion)",
        ]),
        make_code([
            "seg_records = prepare_segmentation_data(splits)",
            "print(f'Segmentation records: {len(seg_records)}')",
        ]),
        make_code([
            "# Save split indices for later reuse",
            "import pickle",
            "split_info = {k: [m[\"filename\"] for m in v] for k, v in splits.items()}",
            "with open(os.path.join(DATA_PROCESSED, 'splits.pkl'), 'wb') as f:",
            "    pickle.dump(split_info, f)",
            "print('Data preparation complete!')",
        ]),
    ]
    write_notebook("01_data_preparation.ipynb", cells)

def notebook_02():
    cells = [
        make_md([
            "# 02 - Image Classification (Week 2)",
            "Training a MobileNetV3 classifier on periapical lesion images (PAI 3, 4, 5).",
        ]),
        make_code([
            "import sys, os, json",
            "# Find project root (works from project root or notebooks/ subdir)",
            "_root = None",
            "for _p in [os.path.abspath('.'), os.path.abspath('..')]:",
            "    if os.path.exists(os.path.join(_p, 'src', 'config.py')):",
            "        _root = _p",
            "        break",
            "if _root:",
            "    sys.path.append(_root)",
            "else:",
            "    raise RuntimeError('Could not find project root (src/config.py not found)')",
            "from src.config import *",
            "from src.train_classification import train_classification, evaluate_classification",
            "from src.visualize import plot_training_history, plot_confusion_matrix, visualize_sample_predictions",
            "import warnings; warnings.filterwarnings('ignore')",
        ]),
        make_code([
            "# Load classification manifest",
            "manifest_path = os.path.join(DATA_PROCESSED, 'classification', 'manifest.json')",
            "with open(manifest_path) as f:",
            "    all_records = json.load(f)",
            "print(f'Loaded {len(all_records)} classification records')",
        ]),
        make_code([
            "# Split records by split field",
            "train_records = [r for r in all_records if r['split'] == 'train']",
            "val_records = [r for r in all_records if r['split'] == 'val']",
            "test_records = [r for r in all_records if r['split'] == 'test']",
            "print(f'Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}')",
        ]),
        make_code([
            "# Train classifier",
            "model, history = train_classification(train_records, val_records)",
        ]),
        make_code([
            "# Plot training history",
            "plot_training_history(history, 'Classification',",
            "    os.path.join(OUTPUTS_DIR, 'classification', 'training_history.png'))",
            "print('Training history plot saved.')",
        ]),
        make_code([
            "# Evaluate on test set",
            "metrics, preds, labels, probs = evaluate_classification(model, test_records)",
        ]),
        make_code([
            "# Plot confusion matrix",
            "plot_confusion_matrix(metrics['confusion_matrix'],",
            "    [CLASS_NAMES[i] for i in [3,4,5]],",
            "    os.path.join(OUTPUTS_DIR, 'classification', 'confusion_matrix.png'))",
            "print('Confusion matrix saved.')",
        ]),
        make_code([
            "# Visualize sample predictions",
            "visualize_sample_predictions(model, test_records,",
            "    output_dir=os.path.join(OUTPUTS_DIR, 'classification'))",
            "print('Sample predictions saved.')",
        ]),
        make_code([
            "# Save metrics as JSON for GUI",
            "with open(os.path.join(OUTPUTS_DIR, 'classification', 'metrics.json'), 'w') as f:",
            "    json.dump(metrics, f, indent=2)",
            "print('Classification complete!')",
        ]),
    ]
    write_notebook("02_classification.ipynb", cells)

def notebook_03():
    cells = [
        make_md([
            "# 03 - Object Detection (Week 3)",
            "Training YOLOv8n for periapical lesion detection with bounding box annotations.",
        ]),
        make_code([
            "import sys, os, json",
            "# Find project root (works from project root or notebooks/ subdir)",
            "_root = None",
            "for _p in [os.path.abspath('.'), os.path.abspath('..')]:",
            "    if os.path.exists(os.path.join(_p, 'src', 'config.py')):",
            "        _root = _p",
            "        break",
            "if _root:",
            "    sys.path.append(_root)",
            "else:",
            "    raise RuntimeError('Could not find project root (src/config.py not found)')",
            "from src.config import *",
            "from src.train_detection import train_detection",
            "import warnings; warnings.filterwarnings('ignore')",
        ]),
        make_code([
            "# Locate the YAML file generated from data preparation",
            "yaml_path = os.path.join(DATA_PROCESSED, 'detection', 'dataset.yaml')",
            "print(f'Using config: {yaml_path}')",
            "import yaml",
            "with open(yaml_path) as f:",
            "    cfg = yaml.safe_load(f)",
            "print(f'Classes: {cfg[\"names\"]}')",
        ]),
        make_code([
            "# Train YOLOv8 detection model",
            "model, metrics = train_detection(yaml_path)",
        ]),
        make_code([
            "# Run prediction on a test image",
            "test_img_dir = os.path.join(DATA_PROCESSED, 'detection', 'images', 'test')",
            "test_images = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg','.png'))]",
            "if test_images:",
            "    img_path = os.path.join(test_img_dir, test_images[0])",
            "    results = model(img_path, device=DEVICE)",
            "    from src.visualize import visualize_detection",
            "    out_path = os.path.join(OUTPUTS_DIR, 'detection', 'sample_detection.png')",
            "    visualize_detection(img_path, results[0], out_path)",
            "    print(f'Saved: {out_path}')",
        ]),
        make_code([
            "# Save metrics",
            "with open(os.path.join(OUTPUTS_DIR, 'detection', 'metrics.json'), 'w') as f:",
            "    json.dump(metrics, f, indent=2)",
            "print('Detection complete!')",
        ]),
    ]
    write_notebook("03_detection.ipynb", cells)

def notebook_04():
    cells = [
        make_md([
            "# 04 - Image Segmentation (Week 4)",
            "Training a UNet for periapical lesion segmentation using box-derived pseudo-masks.",
        ]),
        make_code([
            "import sys, os, json",
            "# Find project root (works from project root or notebooks/ subdir)",
            "_root = None",
            "for _p in [os.path.abspath('.'), os.path.abspath('..')]:",
            "    if os.path.exists(os.path.join(_p, 'src', 'config.py')):",
            "        _root = _p",
            "        break",
            "if _root:",
            "    sys.path.append(_root)",
            "else:",
            "    raise RuntimeError('Could not find project root (src/config.py not found)')",
            "from src.config import *",
            "from src.train_segmentation import train_segmentation, evaluate_segmentation",
            "from src.visualize import plot_training_history",
            "import warnings; warnings.filterwarnings('ignore')",
        ]),
        make_code([
            "# Load segmentation manifest",
            "seg_dir = os.path.join(DATA_PROCESSED, 'segmentation')",
            "import pickle",
            "with open(os.path.join(DATA_PROCESSED, 'splits.pkl'), 'rb') as f:",
            "    split_info = pickle.load(f)",
            "",
            "def load_seg_records(split_name):",
            "    img_dir = os.path.join(seg_dir, 'images', split_name)",
            "    mask_dir = os.path.join(seg_dir, 'masks', split_name)",
            "    records = []",
            "    for fname in os.listdir(img_dir):",
            "        base = os.path.splitext(fname)[0]",
            "        mask_path = os.path.join(mask_dir, base + '.png')",
            "        if os.path.exists(mask_path):",
            "            records.append({",
            "                'image_path': os.path.join(img_dir, fname),",
            "                'mask_path': mask_path,",
            "                'filename': fname,",
            "            })",
            "    return records",
            "",
            "train_records = load_seg_records('train')",
            "val_records = load_seg_records('val')",
            "test_records = load_seg_records('test')",
            "print(f'Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}')",
        ]),
        make_code([
            "# Train segmentation model",
            "model, history = train_segmentation(train_records, val_records)",
        ]),
        make_code([
            "# Plot training history",
            "plot_training_history(history, 'Segmentation',",
            "    os.path.join(OUTPUTS_DIR, 'segmentation', 'training_history.png'))",
            "print('Training history plot saved.')",
        ]),
        make_code([
            "# Evaluate on test set",
            "metrics, pred_masks, true_masks = evaluate_segmentation(model, test_records)",
        ]),
        make_code([
            "# Visualize a sample segmentation",
            "from src.visualize import visualize_segmentation",
            "if test_records:",
            "    rec = test_records[0]",
            "    out_path = os.path.join(OUTPUTS_DIR, 'segmentation', 'sample_segmentation.png')",
            "    visualize_segmentation(rec['image_path'], pred_masks[0].cpu(), out_path)",
            "    print(f'Saved: {out_path}')",
        ]),
        make_code([
            "# Save metrics",
            "with open(os.path.join(OUTPUTS_DIR, 'segmentation', 'metrics.json'), 'w') as f:",
            "    json.dump(metrics, f, indent=2)",
            "print('Segmentation complete!')",
        ]),
    ]
    write_notebook("04_segmentation.ipynb", cells)

if __name__ == "__main__":
    notebook_01()
    notebook_02()
    notebook_03()
    notebook_04()
    print(f"\nAll notebooks created in: {NOTEBOOKS_DIR}")
