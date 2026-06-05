import os
from ultralytics import YOLO
import json

from src.config import *

def train_detection(yaml_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(OUTPUTS_DIR, "detection")
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=yaml_path,
        epochs=EPOCHS_DET,
        imgsz=IMG_SIZE_DET,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=output_dir,
        name="yolo_run",
        exist_ok=True,
        verbose=True,
        workers=0,
    )

    metrics = model.val(data=yaml_path, device=DEVICE)

    results_summary = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nDetection Metrics:")
    print(f"  mAP@50:    {results_summary['mAP50']:.4f}")
    print(f"  mAP@50:95: {results_summary['mAP50_95']:.4f}")
    print(f"  Precision: {results_summary['precision']:.4f}")
    print(f"  Recall:    {results_summary['recall']:.4f}")

    return model, results_summary
