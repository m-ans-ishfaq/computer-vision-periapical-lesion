import sys, os, json, warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import *
from src.data_utils import scan_dataset, build_dataset_manifest, split_dataset, prepare_segmentation_data
from src.train_segmentation import train_segmentation, evaluate_segmentation

original_dir, aug_dir, annot_dir = scan_dataset()
manifest = build_dataset_manifest(original_dir, annot_dir, sample_size=100)
splits = split_dataset(manifest)
seg_records = prepare_segmentation_data(splits)

seg_train = [r for r in seg_records if r['split'] == 'train']
seg_val = [r for r in seg_records if r['split'] == 'val']
seg_test = [r for r in seg_records if r['split'] == 'test']
print(f"Train: {len(seg_train)}, Val: {len(seg_val)}, Test: {len(seg_test)}")

model, history = train_segmentation(seg_train, seg_val)
metrics, _, _ = evaluate_segmentation(model, seg_test)
print(f"Seg n=100: dice={metrics['dice']:.4f}, iou={metrics['iou']:.4f}")

all_results = [
    {"num_samples": 50, "cls_acc": 0.571, "cls_f1": 0.416, "seg_dice": 0.4557, "det_map50": 0.0},
    {"num_samples": 100, "cls_acc": 0.600, "cls_f1": 0.493, "seg_dice": metrics["dice"], "det_map50": 0.0},
]

with open("outputs/classification/metrics.json") as f:
    cls200 = json.load(f)
with open("outputs/segmentation/metrics.json") as f:
    seg200 = json.load(f)
all_results.append({"num_samples": 200, "cls_acc": cls200["accuracy"], "cls_f1": cls200["f1_score"], "seg_dice": seg200["dice"], "det_map50": 3.6e-5})

with open("outputs/scaling_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

for r in all_results:
    print(f"{r['num_samples']:3d} samples: cls_acc={r['cls_acc']:.3f} cls_f1={r['cls_f1']:.3f} seg_dice={r['seg_dice']:.4f}")
print("Saved to outputs/scaling_results.json")
