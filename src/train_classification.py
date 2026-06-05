import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from collections import Counter
from tqdm import tqdm

from src.config import *
from src.models import get_classifier

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def _augment(img):
    w, h = img.size
    if np.random.rand() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    angle = np.random.uniform(-10, 10)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
    if np.random.rand() > 0.5:
        scale = np.random.uniform(0.85, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.BILINEAR)
        pw, ph = w - nw, h - nh
        l, t = np.random.randint(0, pw + 1) if pw > 0 else 0, np.random.randint(0, ph + 1) if ph > 0 else 0
        padded = Image.new('RGB', (w, h), (128, 128, 128))
        padded.paste(img, (l, t))
        img = padded
    bright = np.random.uniform(0.8, 1.2)
    img = img.point(lambda p: min(255, max(0, int(p * bright))))
    return img

def _val_transform(img):
    img = img.resize((IMG_SIZE_CLS, IMG_SIZE_CLS), Image.BILINEAR)
    t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    return (t - MEAN) / STD

class ToothDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        label = rec["label"] - 3
        if self.augment:
            image = _augment(image)
        image = _val_transform(image)
        return image, label

def _device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def train_classification(train_records, val_records, output_dir=None, num_epochs=None, batch_size=None, device=None):
    if output_dir is None:
        output_dir = os.path.join(OUTPUTS_DIR, "classification")
    os.makedirs(output_dir, exist_ok=True)

    device = device or _device()
    batch_size = batch_size or BATCH_SIZE
    epochs = num_epochs or EPOCHS_CLS

    train_dataset = ToothDataset(train_records, augment=True)
    val_dataset = ToothDataset(val_records, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    labels = [r["label"] - 3 for r in train_records]
    counts = Counter(labels)
    weights = [1.0 / counts.get(i, 1) for i in range(NUM_CLASSES)]
    total = sum(weights)
    weights = [w / total * NUM_CLASSES for w in weights]
    weight_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"  Class weights: {dict(zip([3,4,5], [round(w,3) for w in weights]))}")

    model = get_classifier()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    best_epoch = -1
    best_state = None
    patience = 7
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"  * New best! Val Acc={val_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        best_path = os.path.join(output_dir, "classifier.pth")
        torch.save(best_state, best_path)
        print(f"Saved best model from epoch {best_epoch} (Val Acc={best_val_acc:.4f})")
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, "classifier.pth"))

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, history

def evaluate_classification(model, test_records, batch_size=None, device=None):
    device = device or _device()
    batch_size = batch_size or BATCH_SIZE
    dataset = ToothDataset(test_records, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return metrics, all_preds, all_labels, all_probs
