
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "traffic_light_detection_v2"
    / "weights"
    / "best.pt"
)

CLASS_NAMES = {
    0: "red",
    1: "yellow",
    2: "green",
    3: "off",
    4: "wait_on",
}


# --------------------------------------------------
# Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Traffic Light Detection",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Traffic Light Detection")
st.write(
    "Upload an image to detect traffic lights and classify "
    "their signal states."
)


# --------------------------------------------------
# Model loading
# --------------------------------------------------

@st.cache_resource
def load_model(model_path_string):
    return YOLO(model_path_string)


if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Check that the trained weights exist at the path above.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH.resolve()))
except Exception as exc:
    st.error(f"Failed to load the model: {exc}")
    st.stop()


# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------

st.sidebar.header("Inference settings")

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
)

image_size = st.sidebar.selectbox(
    "Inference image size",
    options=[640, 800, 960],
    index=0,
)

device = 0 if torch.cuda.is_available() else "cpu"

st.sidebar.write(f"**Device:** {device}")
st.sidebar.write(f"**CUDA available:** {torch.cuda.is_available()}")
st.sidebar.write(f"**Model:** `{MODEL_PATH}`")

st.sidebar.subheader("Loaded class mapping")
for class_id, class_name in model.names.items():
    st.sidebar.write(f"{class_id}: {class_name}")


# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a traffic-light image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.stop()

try:
    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb_image = np.array(pil_image)
except Exception as exc:
    st.error(f"Could not read the uploaded image: {exc}")
    st.stop()


# --------------------------------------------------
# Original image
# --------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("Original image")
    st.image(rgb_image, width="stretch")

st.caption(
    f"Image dimensions: {rgb_image.shape[1]} × "
    f"{rgb_image.shape[0]} pixels"
)


# --------------------------------------------------
# Detection
# --------------------------------------------------

if st.button("Detect traffic lights", type="primary"):
    with st.spinner("Running inference..."):
        try:
            # IMPORTANT:
            # PIL/NumPy image above is RGB.
            # Ultralytics expects OpenCV-style NumPy images in BGR order.
            # Convert before inference to avoid swapped color channels.
            bgr_image = cv2.cvtColor(
                rgb_image,
                cv2.COLOR_RGB2BGR,
            )

            results = model.predict(
                source=bgr_image,
                conf=confidence,
                imgsz=image_size,
                device=device,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes

            # Ultralytics result.plot() returns a BGR image.
            annotated_bgr = result.plot(
                labels=True,
                conf=True,
            )

            annotated_rgb = cv2.cvtColor(
                annotated_bgr,
                cv2.COLOR_BGR2RGB,
            )

            # --------------------------------------------------
            # Display predictions
            # --------------------------------------------------

            with right:
                st.subheader("Detection result")
                st.image(annotated_rgb, width="stretch")

            st.success(
                f"Detection completed: {len(boxes)} object(s) detected."
            )

            # --------------------------------------------------
            # Diagnostic information
            # --------------------------------------------------

            st.subheader("Inference diagnostics")

            diag1, diag2, diag3 = st.columns(3)

            diag1.metric("Detections", len(boxes))
            diag2.metric("Image size", image_size)
            diag3.metric("Confidence threshold", f"{confidence:.2f}")

            st.write("**Model file:**", str(MODEL_PATH.resolve()))
            st.write("**Runtime device:**", device)
            st.write("**Loaded model classes:**", model.names)

            # --------------------------------------------------
            # Detection details
            # --------------------------------------------------

            st.subheader("Individual predictions")

            if len(boxes) == 0:
                st.warning(
                    "No objects passed the confidence threshold. "
                    "Try lowering the threshold or testing an image "
                    "containing a clearly visible traffic light."
                )
            else:
                prediction_rows = []

                for index, box in enumerate(boxes):
                    class_id = int(box.cls.item())
                    score = float(box.conf.item())
                    coordinates = [
                        round(float(value), 2)
                        for value in box.xyxy[0].cpu().tolist()
                    ]

                    # Use the model's class mapping, not a separate
                    # manually assigned class index.
                    class_name = model.names.get(
                        class_id,
                        f"unknown_{class_id}",
                    )

                    prediction_rows.append(
                        {
                            "Detection": index + 1,
                            "Class ID": class_id,
                            "Predicted state": class_name,
                            "Confidence": round(score, 4),
                            "Bounding box [x1,y1,x2,y2]": str(
                                coordinates
                            ),
                        }
                    )

                st.dataframe(
                    prediction_rows,
                    hide_index=True,
                    width="stretch",
                )

            # --------------------------------------------------
            # Download annotated image
            # --------------------------------------------------

            output_image = Image.fromarray(annotated_rgb)
            buffer = BytesIO()
            output_image.save(buffer, format="PNG")

            st.download_button(
                label="Download annotated image",
                data=buffer.getvalue(),
                file_name="traffic_light_predictions.png",
                mime="image/png",
            )

        except Exception as exc:
            st.error("Inference failed.")
            st.exception(exc)

st.divider()
st.caption(
    "Diagnostic note: confidence scores are model estimates, "
    "not guarantees that a traffic-light state is correct."
)
