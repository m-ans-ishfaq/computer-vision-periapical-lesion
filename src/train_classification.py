import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from tqdm import tqdm

from src.config import *
from src.models import get_classifier

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def default_transform(img):
    img = img.resize((IMG_SIZE_CLS, IMG_SIZE_CLS), Image.BILINEAR)
    t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    return (t - MEAN) / STD

class ToothDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform or default_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        label = rec["label"] - 3
        image = self.transform(image)
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

    train_dataset = ToothDataset(train_records)
    val_dataset = ToothDataset(val_records)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = get_classifier()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
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
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={history['train_loss'][-1]:.4f}, "
              f"Val Loss={history['val_loss'][-1]:.4f}, Val Acc={val_acc:.4f}")

    torch.save(model.state_dict(), os.path.join(output_dir, "classifier.pth"))

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, history

def evaluate_classification(model, test_records, batch_size=None, device=None):
    device = device or _device()
    batch_size = batch_size or BATCH_SIZE
    dataset = ToothDataset(test_records)
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
