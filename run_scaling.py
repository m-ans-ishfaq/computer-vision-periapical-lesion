import sys, os, json, warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import *
from src.data_utils import scan_dataset, build_dataset_manifest, split_dataset, prepare_classification_data, prepare_detection_data, prepare_segmentation_data, get_class_distribution
from src.train_classification import train_classification, evaluate_classification
from src.train_detection import train_detection
from src.train_segmentation import train_segmentation, evaluate_segmentation

results = []

for samples in [50, 100]:
    print("\n" + "="*60)
    print(f"RUNNING WITH NUM_SAMPLES = {samples}")
    print("="*60)
    original_dir, aug_dir, annot_dir = scan_dataset()
    manifest = build_dataset_manifest(original_dir, annot_dir, sample_size=samples)
    dist = get_class_distribution(manifest)
    splits = split_dataset(manifest)
    cls_records = prepare_classification_data(splits)
    det_records, yaml_path = prepare_detection_data(splits)
    seg_records = prepare_segmentation_data(splits)

    cls_train = [r for r in cls_records if r['split'] == 'train']
    cls_val = [r for r in cls_records if r['split'] == 'val']
    cls_test = [r for r in cls_records if r['split'] == 'test']

    print("Classification...")
    cls_model, cls_history = train_classification(cls_train, cls_val)
    cls_metrics, _, _, _ = evaluate_classification(cls_model, cls_test)

    print("Detection...")
    det_model, det_metrics = train_detection(yaml_path)

    seg_train = [r for r in seg_records if r['split'] == 'train']
    seg_val = [r for r in seg_records if r['split'] == 'val']
    seg_test = [r for r in seg_records if r['split'] == 'test']
    print("Segmentation...")
    seg_model, seg_history = train_segmentation(seg_train, seg_val)
    seg_metrics, _, _ = evaluate_segmentation(seg_model, seg_test)

    results.append({
        'num_samples': samples,
        'classification': {
            'accuracy': cls_metrics['accuracy'],
            'precision': cls_metrics['precision'],
            'recall': cls_metrics['recall'],
            'f1_score': cls_metrics['f1_score']
        },
        'detection': {
            'mAP50': det_metrics['mAP50'],
            'mAP50_95': det_metrics['mAP50_95'],
            'precision': det_metrics['precision'],
            'recall': det_metrics['recall']
        },
        'segmentation': {
            'dice': seg_metrics['dice'],
            'iou': seg_metrics['iou']
        }
    })
    print(f"  Cls: acc={cls_metrics['accuracy']:.3f}, f1={cls_metrics['f1_score']:.3f}")
    print(f"  Det: mAP50={det_metrics['mAP50']:.5f}")
    print(f"  Seg: dice={seg_metrics['dice']:.4f}")

print("\n" + "="*60)
print("SCALING RESULTS")
print("="*60)
for r in results:
    print(f"n={r['num_samples']:3d}: cls_acc={r['classification']['accuracy']:.3f} det_map={r['detection']['mAP50']:.5f} seg_dice={r['segmentation']['dice']:.4f}")

with open('outputs/scaling_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved to outputs/scaling_results.json")
