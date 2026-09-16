from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path("results/train/yolo11n_excavator/weights/best.pt")


@st.cache_resource
def load_model(weights: str) -> YOLO:
    return YOLO(weights)


st.set_page_config(page_title="Excavator Detector", page_icon="🏗️", layout="wide")
st.title("Excavator Detector")
st.caption("YOLO11n fine-tuned on the MOCS construction-site dataset")

weights = Path(os.environ.get("EXCAVATOR_WEIGHTS", str(DEFAULT_WEIGHTS)))
with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    st.caption(f"Checkpoint: `{weights}`")

if not weights.is_file():
    st.warning(
        "Model checkpoint not found. Train the detector first, or set the "
        "EXCAVATOR_WEIGHTS environment variable to your best.pt file."
    )
    st.stop()

upload = st.file_uploader("Upload a construction-site image", type=["jpg", "jpeg", "png", "webp"])
if upload is not None:
    image = Image.open(upload).convert("RGB")
    left, right = st.columns(2)
    left.image(image, caption="Input", use_container_width=True)
    with st.spinner("Detecting excavators…"):
        result = load_model(str(weights)).predict(np.asarray(image), conf=confidence, verbose=False)[0]
    annotated = result.plot(conf=True, labels=True, color_mode="class")[:, :, ::-1]
    right.image(annotated, caption="Detections", use_container_width=True)

    confidences = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
    st.subheader(f"{len(confidences)} excavator{'s' if len(confidences) != 1 else ''} detected")
    if confidences:
        st.dataframe(
            {"Detection": list(range(1, len(confidences) + 1)), "Confidence": confidences},
            hide_index=True,
            use_container_width=True,
        )

