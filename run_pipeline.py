import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import *
from src.data_utils import *
from src.models import get_classifier, get_segmenter
from src.train_classification import train_classification, evaluate_classification
from src.train_detection import train_detection
from src.train_segmentation import train_segmentation, evaluate_segmentation
from src.visualize import *

import random, json
random.seed(RANDOM_SEED)

print("=" * 60)
print("STEP 1: Scan dataset")
print("=" * 60)
original_dir, aug_dir, annot_dir = scan_dataset()
print(f'  Original: {original_dir}')
print(f'  Annot:    {annot_dir}')

print("\n" + "=" * 60)
print("STEP 2: Build manifest")
print("=" * 60)
manifest = build_dataset_manifest(original_dir, annot_dir, sample_size=NUM_SAMPLES)
print(f'  Samples: {len(manifest)}')
dist = get_class_distribution(manifest)
print(f'  Distribution: {dist}')

print("\n" + "=" * 60)
print("STEP 3: Split dataset")
print("=" * 60)
splits = split_dataset(manifest)
for k, v in splits.items():
    print(f'  {k}: {len(v)}')

print("\n" + "=" * 60)
print("STEP 4: Prepare classification data")
print("=" * 60)
cls_records = prepare_classification_data(splits)
print(f'  Records: {len(cls_records)}')

print("\n" + "=" * 60)
print("STEP 5: Prepare detection data")
print("=" * 60)
det_records, yaml_path = prepare_detection_data(splits)
print(f'  Records: {len(det_records)}, YAML: {yaml_path}')

print("\n" + "=" * 60)
print("STEP 6: Prepare segmentation data")
print("=" * 60)
seg_records = prepare_segmentation_data(splits)
print(f'  Records: {len(seg_records)}')

print("\n" + "=" * 60)
print("STEP 7: Train classification (MobileNetV3)")
print("=" * 60)
train_rec = [r for r in cls_records if r['split'] == 'train']
val_rec = [r for r in cls_records if r['split'] == 'val']
test_rec = [r for r in cls_records if r['split'] == 'test']
print(f'  Train: {len(train_rec)}, Val: {len(val_rec)}, Test: {len(test_rec)}')
cls_model, cls_history = train_classification(train_rec, val_rec)
plot_training_history(cls_history, "Classification",
    os.path.join(OUTPUTS_DIR, "classification", "training_history.png"))
cls_metrics, _, _, _ = evaluate_classification(cls_model, test_rec)
with open(os.path.join(OUTPUTS_DIR, "classification", "metrics.json"), 'w') as f:
    json.dump(cls_metrics, f, indent=2)

print("\n" + "=" * 60)
print("STEP 8: Train detection (YOLOv8n)")
print("=" * 60)
det_model, det_metrics = train_detection(yaml_path)
with open(os.path.join(OUTPUTS_DIR, "detection", "metrics.json"), 'w') as f:
    json.dump(det_metrics, f, indent=2)

print("\n" + "=" * 60)
print("STEP 9: Train segmentation (Mini-UNet)")
print("=" * 60)
seg_train = [r for r in seg_records if r['split'] == 'train']
seg_val = [r for r in seg_records if r['split'] == 'val']
seg_test = [r for r in seg_records if r['split'] == 'test']
print(f'  Train: {len(seg_train)}, Val: {len(seg_val)}, Test: {len(seg_test)}')
seg_model, seg_history = train_segmentation(seg_train, seg_val)
plot_training_history(seg_history, "Segmentation",
    os.path.join(OUTPUTS_DIR, "segmentation", "training_history.png"))
seg_metrics, _, _ = evaluate_segmentation(seg_model, seg_test)
with open(os.path.join(OUTPUTS_DIR, "segmentation", "metrics.json"), 'w') as f:
    json.dump(seg_metrics, f, indent=2)

print("\n" + "=" * 60)
print("ALL DONE!")
print(f"  Classification: {cls_metrics}")
print(f"  Detection:      {det_metrics}")
print(f"  Segmentation:   {seg_metrics}")
print("=" * 60)
