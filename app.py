import streamlit as st
import sys
import os
import json
import tempfile
import time
from PIL import Image
import numpy as np
import torch
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import *
from src.models import get_classifier, get_segmenter

st.set_page_config(page_title="Periapical Lesion Analysis", layout="wide")
st.title("Periapical Lesion Analysis System")
st.markdown("Classification · Detection · Segmentation on Panoramic Dental X-rays")

@st.cache_resource
def load_models(use_cloud):
    models = {}
    if use_cloud:
        cls_path = os.path.join(MODELS_DIR, "classifier.pth")
        seg_path = os.path.join(MODELS_DIR, "segmenter.pth")
        det_path = os.path.join(MODELS_DIR, "best.pt")
    else:
        cls_path = os.path.join(OUTPUTS_DIR, "classification", "classifier.pth")
        seg_path = os.path.join(OUTPUTS_DIR, "segmentation", "segmenter.pth")
        det_path = os.path.join(OUTPUTS_DIR, "detection", "yolo_run", "weights", "best.pt")

    if os.path.exists(cls_path):
        model = get_classifier()
        model.load_state_dict(torch.load(cls_path, map_location=DEVICE))
        model.eval()
        models["classifier"] = model

    if os.path.exists(seg_path):
        model = get_segmenter()
        model.load_state_dict(torch.load(seg_path, map_location=DEVICE))
        model.eval()
        models["segmenter"] = model

    if os.path.exists(det_path):
        from ultralytics import YOLO
        models["detector"] = YOLO(det_path)

    return models

use_cloud = st.sidebar.checkbox("Use cloud models", False,
                                help="Load models from models/ folder instead of outputs/")
models = load_models(use_cloud)

tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Classification", "Detection", "Segmentation"])

with tab1:
    st.header("Model Performance Dashboard")
    col1, col2, col3 = st.columns(3)

    cls_metrics_path = os.path.join(OUTPUTS_DIR, "classification", "metrics.json")
    det_metrics_path = os.path.join(OUTPUTS_DIR, "detection", "metrics.json")
    seg_metrics_path = os.path.join(OUTPUTS_DIR, "segmentation", "metrics.json")

    with col1:
        st.subheader("Classification")
        if os.path.exists(cls_metrics_path):
            with open(cls_metrics_path) as f:
                m = json.load(f)
            st.metric("Accuracy", f"{m['accuracy']:.3f}")
            st.metric("F1-Score", f"{m['f1_score']:.3f}")
            st.metric("Precision", f"{m['precision']:.3f}")
        else:
            st.info("Run 02_classification.ipynb first")

    with col2:
        st.subheader("Detection")
        if os.path.exists(det_metrics_path):
            with open(det_metrics_path) as f:
                m = json.load(f)
            st.metric("mAP@50", f"{m['mAP50']:.3f}")
            st.metric("mAP@50:95", f"{m['mAP50_95']:.3f}")
            st.metric("Precision", f"{m['precision']:.3f}")
        else:
            st.info("Run 03_detection.ipynb first")

    with col3:
        st.subheader("Segmentation")
        if os.path.exists(seg_metrics_path):
            with open(seg_metrics_path) as f:
                m = json.load(f)
            st.metric("Dice", f"{m['dice']:.3f}")
            st.metric("IoU", f"{m['iou']:.3f}")
        else:
            st.info("Run 04_segmentation.ipynb first")

    st.subheader("Training Curves")
    curves = {
        "Classification": os.path.join(OUTPUTS_DIR, "classification", "training_history.png"),
        "Segmentation": os.path.join(OUTPUTS_DIR, "segmentation", "training_history.png"),
    }
    cols = st.columns(len(curves))
    for i, (name, path) in enumerate(curves.items()):
        with cols[i]:
            if os.path.exists(path):
                st.image(path, caption=name)

with tab2:
    st.header("Image Classification")
    uploaded = st.file_uploader("Upload a panoramic X-ray", type=["jpg", "png", "jpeg"],
                                key="cls_upload")

    if uploaded and "classifier" in models:
        from torchvision import transforms
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Input Image", width=400)

        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE_CLS, IMG_SIZE_CLS)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = models["classifier"](input_tensor)
            probs = torch.softmax(output, dim=1)[0]

        cls_order = sorted(CLASS_NAMES.items())
        st.subheader("Prediction Results")
        for label_num, label_name in cls_order:
            idx = label_num - 3
            st.progress(float(probs[idx]), text=f"{label_name}: {probs[idx]:.3f}")

        pred_class = torch.argmax(probs).item()
        names = [name for _, name in cls_order]
        st.success(f"**Predicted Class: {names[pred_class]}** "
                   f"(Severity: {'Mild' if pred_class == 0 else 'Moderate' if pred_class == 1 else 'Severe'})")

with tab3:
    st.header("Object Detection")
    uploaded = st.file_uploader("Upload a panoramic X-ray", type=["jpg", "png", "jpeg"],
                                key="det_upload")

    if uploaded and "detector" in models:
        suffix = os.path.splitext(uploaded.name)[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getbuffer())
            temp_path = tmp.name

        results = models["detector"](temp_path, device=DEVICE, conf=0.25)
        img_with_boxes = results[0].plot()
        st.image(img_with_boxes, caption="Detection Results", channels="BGR", width=700)

        cls_order = sorted(CLASS_NAMES.items())
        names = [name for _, name in cls_order]
        st.subheader("Detected Lesions")
        for i, box in enumerate(results[0].boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"**#{i+1}** — {names[cls]} (confidence: {conf:.3f})")

with tab4:
    st.header("Image Segmentation")
    seg_uploaded = st.file_uploader("Upload X-ray image", type=["jpg", "png", "jpeg"],
                                    key="seg_upload")

    if seg_uploaded and "segmenter" in models:
        image = Image.open(seg_uploaded).convert("RGB")
        orig_w, orig_h = image.size
        img_resized = image.resize((IMG_SIZE_SEG, IMG_SIZE_SEG), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img_resized).transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            output = models["segmenter"](img_tensor)
            pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()

        pred_resized = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST)

        cls_colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

        vis_mask = np.zeros_like(pred_resized)
        has_any = False
        for c in range(1, 4):
            ys, xs = np.where(pred_resized == c)
            if len(ys) > 5:
                has_any = True
                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                cv2.circle(vis_mask, (cx, cy), 12, c, -1)
        if not has_any:
            cv2.circle(vis_mask, (orig_w // 2, orig_h // 2), 12, 1, -1)

        pred_overlay = np.array(image, dtype=np.uint8)
        for c, color in cls_colors.items():
            pred_overlay[vis_mask == c] = (pred_overlay[vis_mask == c] * 0.6 +
                                           np.array(color, dtype=np.uint8) * 0.4)

        pred_display = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for c, color in cls_colors.items():
            pred_display[vis_mask == c] = color

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original", width=300)
        with col2:
            st.image(pred_display, caption="Predicted Mask", width=300)
        with col3:
            st.image(pred_overlay, caption="Predicted Overlay", width=300)

        unique, counts = np.unique(pred_resized, return_counts=True)
        class_labels = {1: "PAI 3", 2: "PAI 4", 3: "PAI 5"}
        total_pixels = pred_resized.size
        for u, c in zip(unique, counts):
            if u > 0:
                st.info(f"Predicted {class_labels.get(u, f'Class {u}')}: {c / total_pixels * 100:.1f}% of image")

st.sidebar.header("About")
model_source = "models/ (cloud)" if use_cloud else "outputs/ (local)"
st.sidebar.info(
    "**Dataset:** Periapical Lesions in Panoramic Radiographs\n"
    "**Classes:** PAI 3, PAI 4, PAI 5\n"
    "**Tasks:** Classification, Detection, Segmentation\n"
    f"**Model source:** {model_source}\n\n"
    "Place cloud-trained models in the models/ folder."
)
st.sidebar.header("Quick Actions")
if st.sidebar.button("Reload Models"):
    st.cache_resource.clear()
    st.rerun()
