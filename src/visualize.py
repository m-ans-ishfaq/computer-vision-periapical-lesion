import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import torch
from PIL import Image
from src.config import OUTPUTS_DIR, CLASS_NAMES

def plot_training_history(history, title, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title(f"{title} - Loss")

    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="Val Accuracy", color="green")
        axes[1].set_ylabel("Accuracy")
    elif "val_dice" in history:
        axes[1].plot(history["val_dice"], label="Val Dice", color="green")
        axes[1].plot(history["val_iou"], label="Val IoU", color="orange")
        axes[1].set_ylabel("Score")

    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].set_title(f"{title} - Metrics")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(cm, class_names, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def visualize_detection(image_path, results, output_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{CLASS_NAMES[cls + 3]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls % len(colors)], 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls % len(colors)], 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Detection Results")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def visualize_segmentation(image_path, mask_tensor, output_path, num_classes=4):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (mask_tensor.shape[1], mask_tensor.shape[0]))

    mask_np = mask_tensor.numpy() if hasattr(mask_tensor, "numpy") else mask_tensor
    overlay = np.zeros_like(img, dtype=np.uint8)
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for c in range(1, num_classes):
        overlay[mask_np == c] = colors[c]

    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(mask_np, cmap="tab10", vmin=0, vmax=num_classes - 1)
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")
    axes[2].imshow(blended)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def visualize_sample_predictions(model, test_records, num_samples=4, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(OUTPUTS_DIR, "samples")
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    def transform(img):
        img = img.resize((224, 224), Image.BILINEAR)
        t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        return (t - MEAN) / STD

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(min(num_samples, len(test_records))):
        rec = test_records[i]
        image = Image.open(rec["image_path"]).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)[0]
            pred_class = torch.argmax(prob).item()

        label_name = CLASS_NAMES.get(rec["label"], "Unknown")
        pred_name = CLASS_NAMES.get(pred_class + 3, "Unknown")

        axes[i].imshow(image)
        axes[i].set_title(f"True: {label_name}\nPred: {pred_name} ({prob[pred_class]:.2f})")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_samples.png"), dpi=150)
    plt.close()
