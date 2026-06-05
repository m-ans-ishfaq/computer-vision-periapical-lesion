import os
import torch
from ultralytics import YOLO
import json

from src.config import *

def _device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def train_detection(yaml_path, output_dir=None, epochs=None, batch_size=None, device=None):
    if output_dir is None:
        output_dir = os.path.join(OUTPUTS_DIR, "detection")
    os.makedirs(output_dir, exist_ok=True)

    device = device or _device()
    batch_size = batch_size or BATCH_SIZE
    epochs = epochs or EPOCHS_DET

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=IMG_SIZE_DET,
        batch=batch_size,
        device=device,
        project=output_dir,
        name="yolo_run",
        exist_ok=True,
        verbose=True,
        workers=2,
    )

    metrics = model.val(data=yaml_path, device=device)

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
