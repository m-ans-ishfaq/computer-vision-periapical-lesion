import sys, os, json, warnings, random
import torch
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import *
from src.data_utils import scan_dataset, build_dataset_manifest, split_dataset, prepare_segmentation_data
from src.train_segmentation import train_segmentation, evaluate_segmentation
from src.visualize import plot_training_history

OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "segmentation_deep")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_PCT = 0.05
DEEP_EPOCHS = 50
random.seed(RANDOM_SEED)

resume_path = os.path.join(OUTPUT_DIR, "checkpoint.pth")
if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location="cpu")
    completed = ckpt["epoch"]
    remaining = DEEP_EPOCHS - completed
    if remaining <= 0:
        print(f"Training already complete ({completed}/{DEEP_EPOCHS} epochs).")
        print(f"Results in: {OUTPUT_DIR}")
        sys.exit(0)
    print(f"RESUMING from epoch {completed} ({remaining} epochs remaining)")
    resume_arg = resume_path
else:
    resume_arg = None

print("=" * 60)
print("DEEP SEGMENTATION TRAINING")
print(f"  5% dataset | {DEEP_EPOCHS} epochs")
print("=" * 60)

original_dir, aug_dir, annot_dir = scan_dataset()
full_size = len([f for f in os.listdir(original_dir) if f.endswith('.jpg')])
target_samples = int(full_size * SAMPLE_PCT)
print(f"  Full dataset: ~{full_size} images")
print(f"  Using 5%:     ~{target_samples} samples")

manifest = build_dataset_manifest(original_dir, annot_dir, sample_size=target_samples)
print(f"  Manifest: {len(manifest)} records")

splits = split_dataset(manifest)
seg_records = prepare_segmentation_data(splits)

seg_train = [r for r in seg_records if r['split'] == 'train']
seg_val = [r for r in seg_records if r['split'] == 'val']
seg_test = [r for r in seg_records if r['split'] == 'test']
print(f"  Train: {len(seg_train)}, Val: {len(seg_val)}, Test: {len(seg_test)}")

model, history = train_segmentation(
    seg_train, seg_val,
    output_dir=OUTPUT_DIR,
    num_epochs=DEEP_EPOCHS,
    resume_from=resume_arg
)

plot_training_history(history, "Segmentation (Deep)",
    os.path.join(OUTPUT_DIR, "training_history.png"))

metrics, _, _ = evaluate_segmentation(model, seg_test)
with open(os.path.join(OUTPUT_DIR, "metrics.json"), 'w') as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "history.json"), 'w') as f:
    json.dump(history, f, indent=2)

config_summary = {
    "epochs": DEEP_EPOCHS,
    "sample_pct": SAMPLE_PCT,
    "total_samples": len(manifest),
    "train_samples": len(seg_train),
    "val_samples": len(seg_val),
    "test_samples": len(seg_test),
}
with open(os.path.join(OUTPUT_DIR, "config.json"), 'w') as f:
    json.dump(config_summary, f, indent=2)

print("\n" + "=" * 60)
print(f"DONE! Metrics: {metrics}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 60)
