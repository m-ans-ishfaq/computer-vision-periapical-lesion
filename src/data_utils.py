import os
import xml.etree.ElementTree as ET
import random
import shutil
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import json

import yaml
from src.config import *

LABEL_MAP = {"PAI 3": 3, "PAI 4": 4, "PAI 5": 5, "3": 3, "4": 4, "5": 5}
VALID_LABELS = {3, 4, 5}

def parse_labelme_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename_el = root.find("filename")
    if filename_el is None or filename_el.text is None:
        return None
    filename = filename_el.text
    size_el = root.find("size")
    if size_el is None:
        return None
    width_el = size_el.find("width")
    height_el = size_el.find("height")
    if width_el is None or height_el is None:
        return None
    width = int(width_el.text)
    height = int(height_el.text)

    objects = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or name_el.text is None:
            continue
        label_text = name_el.text.strip()
        label_num = LABEL_MAP.get(label_text)
        if label_num is None:
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin_el, ymin_el = bbox.find("xmin"), bbox.find("ymin")
        xmax_el, ymax_el = bbox.find("xmax"), bbox.find("ymax")
        if any(el is None for el in [xmin_el, ymin_el, xmax_el, ymax_el]):
            continue
        xmin = int(float(xmin_el.text))
        ymin = int(float(ymin_el.text))
        xmax = int(float(xmax_el.text))
        ymax = int(float(ymax_el.text))
        objects.append({"label": label_num, "bbox": [xmin, ymin, xmax, ymax]})

    return {"filename": filename, "width": width, "height": height, "objects": objects}

def scan_dataset(data_dir=None):
    if data_dir is None:
        data_dir = DATA_RAW

    if not os.path.exists(data_dir):
        return None, None, None

    original_dir, aug_dir, annot_dir = None, None, None

    for root, dirs, _ in os.walk(data_dir):
        for d in dirs:
            lower = d.lower()
            if "original" in lower and "image" in lower:
                original_dir = os.path.join(root, d)
            elif "augment" in lower:
                aug_dir = os.path.join(root, d)
            elif "annot" in lower:
                annot_dir = os.path.join(root, d)

    return original_dir, aug_dir, annot_dir

def build_dataset_manifest(original_dir, annot_dir, sample_size=None):
    annotations = {}
    if os.path.exists(annot_dir):
        for f in os.listdir(annot_dir):
            if f.endswith(".xml"):
                xml_path = os.path.join(annot_dir, f)
                try:
                    parsed = parse_labelme_xml(xml_path)
                    if parsed is not None:
                        annotations[f] = parsed
                except Exception as e:
                    print(f"  Warning: skipping {xml_path}: {e}")

    manifest = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    if os.path.exists(original_dir):
        for f in os.listdir(original_dir):
            if any(f.lower().endswith(ext) for ext in image_extensions):
                img_path = os.path.join(original_dir, f)
                xml_name = os.path.splitext(f)[0] + ".xml"
                annot = annotations.get(xml_name, None)
                manifest.append({
                    "image_path": img_path,
                    "filename": f,
                    "annotation": annot,
                    "source": "original"
                })

    if not manifest:
        print("  Warning: no images found in dataset!")
        return manifest

    if sample_size and len(manifest) > sample_size:
        manifest = random.sample(manifest, sample_size)

    return manifest

def get_image_level_label(annot):
    if annot is None or len(annot["objects"]) == 0:
        return None
    labels = [obj["label"] for obj in annot["objects"] if obj["label"] in VALID_LABELS]
    if not labels:
        return None
    return max(labels)

def split_dataset(manifest, val_split=VAL_SPLIT, test_split=TEST_SPLIT, seed=RANDOM_SEED):
    random.seed(seed)
    indices = list(range(len(manifest)))
    random.shuffle(indices)

    n_val = int(len(indices) * val_split)
    n_test = int(len(indices) * test_split)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    return {
        "train": [manifest[i] for i in train_idx],
        "val": [manifest[i] for i in val_idx],
        "test": [manifest[i] for i in test_idx]
    }

def prepare_classification_data(splits, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(DATA_PROCESSED, "classification")

    for split_name, items in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        for label_name in CLASS_NAMES.values():
            os.makedirs(os.path.join(split_dir, label_name), exist_ok=True)

    records = []
    for split_name, items in splits.items():
        for item in items:
            label = get_image_level_label(item["annotation"])
            if label is None:
                continue
            label_name = CLASS_NAMES[label]
            dest = os.path.join(output_dir, split_name, label_name, item["filename"])
            shutil.copy2(item["image_path"], dest)
            records.append({
                "filename": item["filename"],
                "split": split_name,
                "label": label,
                "label_name": label_name,
                "image_path": dest
            })

    json_path = os.path.join(output_dir, "manifest.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    return records

def prepare_detection_data(splits, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(DATA_PROCESSED, "detection")

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    for split_name in ["train", "val", "test"]:
        os.makedirs(os.path.join(images_dir, split_name), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split_name), exist_ok=True)

    yaml_content = {
        "path": output_dir,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": NUM_CLASSES,
        "names": [CLASS_NAMES[i] for i in [3, 4, 5]]
    }

    records = []
    for split_name, items in splits.items():
        for item in items:
            if item["annotation"] is None or len(item["annotation"]["objects"]) == 0:
                continue
            img = cv2.imread(item["image_path"])
            if img is None:
                continue
            h, w = img.shape[:2]

            dest_img = os.path.join(images_dir, split_name, item["filename"])
            shutil.copy2(item["image_path"], dest_img)

            label_path = os.path.join(labels_dir, split_name,
                                      os.path.splitext(item["filename"])[0] + ".txt")
            with open(label_path, "w") as f:
                for obj in item["annotation"]["objects"]:
                    class_id = obj["label"] - 3
                    xmin, ymin, xmax, ymax = obj["bbox"]
                    x_center = ((xmin + xmax) / 2) / w
                    y_center = ((ymin + ymax) / 2) / h
                    bw = (xmax - xmin) / w
                    bh = (ymax - ymin) / h
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

            records.append({
                "filename": item["filename"],
                "split": split_name,
                "image_path": dest_img,
                "label_path": label_path
            })

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    return records, yaml_path

def prepare_segmentation_data(splits, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(DATA_PROCESSED, "segmentation")

    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    for split_name in ["train", "val", "test"]:
        os.makedirs(os.path.join(images_dir, split_name), exist_ok=True)
        os.makedirs(os.path.join(masks_dir, split_name), exist_ok=True)

    records = []
    for split_name, items in splits.items():
        for item in items:
            if item["annotation"] is None or len(item["annotation"]["objects"]) == 0:
                continue
            img = cv2.imread(item["image_path"])
            if img is None:
                continue
            h, w = img.shape[:2]

            mask = np.zeros((h, w), dtype=np.uint8)
            for obj in item["annotation"]["objects"]:
                xmin, ymin, xmax, ymax = obj["bbox"]
                seg_class = {3: 1, 4: 2, 5: 3}.get(obj["label"])
                if seg_class is None:
                    continue
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), seg_class, -1)

            dest_img = os.path.join(images_dir, split_name, item["filename"])
            dest_mask = os.path.join(masks_dir, split_name,
                                     os.path.splitext(item["filename"])[0] + ".png")
            shutil.copy2(item["image_path"], dest_img)
            cv2.imwrite(dest_mask, mask)

            records.append({
                "filename": item["filename"],
                "split": split_name,
                "image_path": dest_img,
                "mask_path": dest_mask,
                "height": h,
                "width": w
            })

    return records

def get_class_distribution(manifest):
    dist = defaultdict(int)
    for item in manifest:
        label = get_image_level_label(item["annotation"])
        if label is not None:
            name = CLASS_NAMES.get(label)
            if name:
                dist[name] += 1
    return dict(dist)
