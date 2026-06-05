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
import torchvision.transforms as T

from src.config import *
from src.models import get_classifier

_train_tfm = lambda sz: T.Compose([
    T.Resize((sz, sz)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_val_tfm = lambda sz: T.Compose([
    T.Resize((sz, sz)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ToothDataset(Dataset):
    def __init__(self, records, augment=False, cache=False, imgsz=224):
        self.records = records
        self._tfm = _train_tfm(imgsz) if augment else _val_tfm(imgsz)
        self.augment = augment
        self.cache = cache
        self._imgs = None
        if cache:
            self._imgs = []
            for rec in tqdm(records, desc="Caching images"):
                img = np.array(Image.open(rec["image_path"]).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR))
                self._imgs.append(img)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        if self.cache:
            image = Image.fromarray(self._imgs[idx].copy())
        else:
            image = Image.open(rec["image_path"]).convert("RGB")
        label = rec["label"] - 3
        return self._tfm(image), label

def _device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def train_classification(train_records, val_records, output_dir=None, num_epochs=None, batch_size=None, device=None, backbone="resnet50", imgsz=384):
    if output_dir is None:
        output_dir = os.path.join(OUTPUTS_DIR, "classification")
    os.makedirs(output_dir, exist_ok=True)

    device = device or _device()
    batch_size = batch_size or BATCH_SIZE
    epochs = num_epochs or EPOCHS_CLS

    print("Caching datasets in RAM...")
    train_dataset = ToothDataset(train_records, augment=True, cache=True, imgsz=imgsz)
    val_dataset = ToothDataset(val_records, augment=False, cache=True, imgsz=imgsz)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    from src.models import BACKBONES
    _, head_attr, _ = BACKBONES[backbone]

    model = get_classifier(backbone=backbone)
    model = model.to(device)

    # Freeze backbone for first 2 epochs (let head learn first)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if head_attr.replace(".0", "") in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    best_epoch = -1
    best_state = None
    patience = 7
    no_improve = 0
    freeze_epochs = 2

    for epoch in range(epochs):
        # Unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs:
            for p in backbone_params:
                p.requires_grad = True
            print(f"  Unfreezing backbone at epoch {epoch+1}")

        if epoch < freeze_epochs:
            for p in backbone_params:
                p.requires_grad = False

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

def evaluate_classification(model, test_records, batch_size=None, device=None, imgsz=384):
    device = device or _device()
    batch_size = batch_size or BATCH_SIZE
    dataset = ToothDataset(test_records, augment=False, imgsz=imgsz)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

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
