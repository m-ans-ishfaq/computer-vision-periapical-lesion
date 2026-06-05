import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json
from tqdm import tqdm

from src.config import *
from src.models import get_segmenter

class SegDataset(Dataset):
    def __init__(self, records, img_size=IMG_SIZE_SEG):
        self.records = records
        self.img_size = img_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        mask = Image.open(rec["mask_path"])

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        img_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64)).long()

        return img_tensor, mask_tensor

def compute_dice(pred, target, num_classes, smooth=1e-6):
    pred_one_hot = torch.nn.functional.one_hot(pred, num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = torch.nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3))
    union = pred_one_hot.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def compute_iou(pred, target, num_classes, smooth=1e-6):
    pred_one_hot = torch.nn.functional.one_hot(pred, num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = torch.nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3))
    union = pred_one_hot.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def train_segmentation(train_records, val_records, output_dir=None, num_epochs=None, resume_from=None):
    if output_dir is None:
        output_dir = os.path.join(OUTPUTS_DIR, "segmentation")
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = SegDataset(train_records)
    val_dataset = SegDataset(val_records)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    history_path = os.path.join(output_dir, "history.json")

    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=DEVICE)
        model = get_segmenter()
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        history = ckpt.get("history", history)
        print(f"  Resumed from epoch {start_epoch}")
    else:
        model = get_segmenter()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    epochs = num_epochs if num_epochs is not None else EPOCHS_SEG

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Seg Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        all_preds, all_masks = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds)
                all_masks.append(masks)

        if not all_preds:
            print("  Warning: empty validation loader, skipping metrics")
            continue
        all_preds = torch.cat(all_preds)
        all_masks = torch.cat(all_masks)

        dice = compute_dice(all_preds, all_masks, NUM_CLASSES + 1)
        iou = compute_iou(all_preds, all_masks, NUM_CLASSES + 1)

        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_dice"].append(dice)
        history["val_iou"].append(iou)

        print(f"Epoch {epoch+1}: Train Loss={history['train_loss'][-1]:.4f}, "
              f"Val Loss={history['val_loss'][-1]:.4f}, Dice={dice:.4f}, IoU={iou:.4f}")

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }, checkpoint_path)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    torch.save(model.state_dict(), os.path.join(output_dir, "segmenter.pth"))

    return model, history

def evaluate_segmentation(model, test_records):
    dataset = SegDataset(test_records)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model.eval()
    all_preds, all_masks = [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())

    if not all_preds:
        return {"dice": 0.0, "iou": 0.0}, None, None
    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)

    dice = compute_dice(all_preds, all_masks, NUM_CLASSES + 1)
    iou = compute_iou(all_preds, all_masks, NUM_CLASSES + 1)

    metrics = {"dice": float(dice), "iou": float(iou)}
    print(f"\nSegmentation Metrics:")
    print(f"  Dice: {dice:.4f}")
    print(f"  IoU:  {iou:.4f}")

    return metrics, all_preds, all_masks
