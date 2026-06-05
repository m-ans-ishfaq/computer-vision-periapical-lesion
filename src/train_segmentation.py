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

def _device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class SegDataset(Dataset):
    def __init__(self, records, img_size=384, augment=False, cache=False):
        self.records = records
        self.img_size = img_size
        self.augment = augment
        self.cache = cache
        self._imgs = None
        self._masks = None
        if cache:
            self._imgs, self._masks = [], []
            for rec in tqdm(records, desc="Caching"):
                img = Image.open(rec["image_path"]).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
                mask = Image.open(rec["mask_path"]).resize((img_size, img_size), Image.NEAREST)
                self._imgs.append(np.array(img, dtype=np.uint8))
                self._masks.append(np.array(mask, dtype=np.int64))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if self.cache:
            image = self._imgs[idx].copy()
            mask = self._masks[idx].copy()
        else:
            rec = self.records[idx]
            image = np.array(Image.open(rec["image_path"]).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR))
            mask = np.array(Image.open(rec["mask_path"]).resize((self.img_size, self.img_size), Image.NEAREST), dtype=np.int64)

        if self.augment:
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            angle = np.random.uniform(-10, 10)
            if abs(angle) > 0.5:
                from PIL import Image as PILImage
                img_pil = PILImage.fromarray(image)
                msk_pil = PILImage.fromarray(mask.astype(np.uint8))
                image = np.array(img_pil.rotate(angle, resample=PILImage.BILINEAR, fill=0))
                mask = np.array(msk_pil.rotate(angle, resample=PILImage.NEAREST, fill=0), dtype=np.int64)

        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask).long()
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

class ComboLoss(nn.Module):
    def __init__(self, num_classes, dice_weight=0.5, ce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

def train_segmentation(train_records, val_records, output_dir=None, num_epochs=None, batch_size=None, device=None, resume_from=None, img_size=384):
    if output_dir is None:
        output_dir = os.path.join(OUTPUTS_DIR, "segmentation")
    os.makedirs(output_dir, exist_ok=True)

    device = device or _device()
    batch_size = batch_size or BATCH_SIZE
    epochs = num_epochs or EPOCHS_SEG

    print("Caching datasets in RAM...")
    train_dataset = SegDataset(train_records, img_size=img_size, augment=True, cache=True)
    val_dataset = SegDataset(val_records, img_size=img_size, augment=False, cache=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    num_classes = NUM_CLASSES + 1

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device)
        model = get_segmenter(img_size=img_size)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder")]
        dec_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
        optimizer = torch.optim.AdamW([
            {'params': enc_params, 'lr': 1e-4},
            {'params': dec_params, 'lr': 1e-3},
        ], weight_decay=1e-4)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        history = ckpt.get("history", {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []})
        best_val_dice = ckpt.get("best_val_dice", 0.0)
        print(f"  Resumed from epoch {start_epoch}")
    else:
        model = get_segmenter(img_size=img_size)
        model = model.to(device)
        enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder")]
        dec_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
        optimizer = torch.optim.AdamW([
            {'params': enc_params, 'lr': 1e-4},
            {'params': dec_params, 'lr': 1e-3},
        ], weight_decay=1e-4)
        start_epoch = 0
        history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
        best_val_dice = 0.0

    criterion = ComboLoss(num_classes=num_classes, dice_weight=0.5, ce_weight=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = None
    best_epoch = -1
    patience = 10
    no_improve = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Seg Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        all_preds, all_masks = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
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

        dice = compute_dice(all_preds, all_masks, num_classes)
        iou = compute_iou(all_preds, all_masks, num_classes)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(dice)
        history["val_iou"].append(iou)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Dice={dice:.4f}, IoU={iou:.4f}, LR={current_lr:.2e}")

        if dice > best_val_dice:
            best_val_dice = dice
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"  * New best! Dice={dice:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_val_dice": best_val_dice,
        }, os.path.join(output_dir, "checkpoint.pth"))

    if best_state is not None:
        model.load_state_dict(best_state)
        best_path = os.path.join(output_dir, "segmenter.pth")
        torch.save(best_state, best_path)
        print(f"Saved best model from epoch {best_epoch} (Dice={best_val_dice:.4f})")
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, "segmenter.pth"))

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, history

def evaluate_segmentation(model, test_records, batch_size=None, device=None, img_size=384):
    device = device or _device()
    batch_size = batch_size or BATCH_SIZE
    dataset = SegDataset(test_records, img_size=img_size, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    model.eval()
    all_preds, all_masks = [], []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())

    if not all_preds:
        return {"dice": 0.0, "iou": 0.0}, None, None
    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)

    num_classes = NUM_CLASSES + 1
    dice = compute_dice(all_preds, all_masks, num_classes)
    iou = compute_iou(all_preds, all_masks, num_classes)

    metrics = {"dice": float(dice), "iou": float(iou)}
    print(f"\nSegmentation Metrics:")
    print(f"  Dice: {dice:.4f}")
    print(f"  IoU:  {iou:.4f}")

    return metrics, all_preds, all_masks
