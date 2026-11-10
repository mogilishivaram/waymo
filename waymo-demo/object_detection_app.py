
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

VEHICLE_MODEL_PATH = (
    ROOT
    / "use_cases"
    / "object_tracking"
    / "models"
    / "waymo_vehicle_baseline-3"
    / "weights"
    / "best.pt"
)

TRAFFIC_MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "traffic_light_detection_v2"
    / "weights"
    / "best.pt"
)


# --------------------------------------------------
# Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="RoadVision AI",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 RoadVision AI")
st.caption(
    "Upload a road image and detect vehicles, "
    "traffic lights, or both."
)


# --------------------------------------------------
# Load existing trained models
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {path}"
        )

    return YOLO(str(path))


# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------

st.sidebar.header("Detection Settings")

model_choice = st.sidebar.selectbox(
    "Select detection model",
    [
        "Vehicle / Object Detection",
        "Traffic-Light Detection",
        "Both Models",
    ],
)

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.35,
    step=0.05,
)

image_size = st.sidebar.selectbox(
    "Inference image size",
    [640, 960, 1280],
    index=0,
)

st.sidebar.info(
    "Uses your existing trained models. "
    "No retraining is required."
)


# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a raw road image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is None:
    st.info(
        "Upload an image to begin object detection."
    )
    st.stop()

try:
    original_image = Image.open(uploaded_file).convert("RGB")
    original_array = np.array(original_image)

except Exception as error:
    st.error(f"Could not read the image: {error}")
    st.stop()


# --------------------------------------------------
# Display original image
# --------------------------------------------------

st.subheader("Input Image")
st.image(
    original_image,
    caption=uploaded_file.name,
    use_container_width=True,
)


# --------------------------------------------------
# Run predictions
# --------------------------------------------------

if st.button(
    "🔍 Detect Objects",
    type="primary",
    use_container_width=True,
):

    selected_models = []

    if model_choice in [
        "Vehicle / Object Detection",
        "Both Models",
    ]:
        selected_models.append(
            ("Vehicle", VEHICLE_MODEL_PATH)
        )

    if model_choice in [
        "Traffic-Light Detection",
        "Both Models",
    ]:
        selected_models.append(
            ("Traffic Light", TRAFFIC_MODEL_PATH)
        )

    missing_models = [
        str(path)
        for _, path in selected_models
        if not path.is_file()
    ]

    if missing_models:
        st.error(
            "The following model file(s) were not found:"
        )

        for path in missing_models:
            st.code(path)

        st.stop()

    # OpenCV drawing functions use BGR color values.
    annotated_image = cv2.cvtColor(
        original_array,
        cv2.COLOR_RGB2BGR,
    )

    all_detections = []

    try:
        with st.spinner(
            "Running detection models..."
        ):

            for model_label, model_path in selected_models:

                model = load_model(str(model_path))

                results = model.predict(
                    source=original_array,
                    conf=confidence,
                    imgsz=image_size,
                    verbose=False,
                )

                result = results[0]

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    class_id = int(
                        box.cls[0].item()
                    )

                    score = float(
                        box.conf[0].item()
                    )

                    if isinstance(model.names, dict):
                        class_name = model.names.get(
                            class_id,
                            str(class_id),
                        )
                    else:
                        class_name = model.names[class_id]

                    # Different colors for the two models.
                    if model_label == "Vehicle":
                        color = (50, 220, 50)
                    else:
                        color = (0, 190, 255)

                    label = (
                        f"{model_label}: {class_name} "
                        f"{score:.2f}"
                    )

                    # Draw bounding box.
                    cv2.rectangle(
                        annotated_image,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    # Draw label background.
                    (text_width, text_height), baseline = (
                        cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            2,
                        )
                    )

                    label_top = max(
                        0,
                        y1 - text_height - baseline - 8,
                    )

                    cv2.rectangle(
                        annotated_image,
                        (x1, label_top),
                        (
                            x1 + text_width + 8,
                            y1,
                        ),
                        color,
                        -1,
                    )

                    cv2.putText(
                        annotated_image,
                        label,
                        (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    all_detections.append({
                        "Model": model_label,
                        "Object": class_name,
                        "Confidence": round(score, 3),
                        "X1": x1,
                        "Y1": y1,
                        "X2": x2,
                        "Y2": y2,
                    })

        # Convert BGR back to RGB for display.
        prediction_rgb = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_BGR2RGB,
        )

        st.session_state["prediction_image"] = prediction_rgb
        st.session_state["detections"] = all_detections
        st.session_state["source_name"] = uploaded_file.name

    except Exception as error:
        st.error(
            f"Detection failed: {error}"
        )
        st.exception(error)


# --------------------------------------------------
# Display prediction results
# --------------------------------------------------

if (
    "prediction_image" in st.session_state
    and st.session_state.get("source_name")
    == uploaded_file.name
):

    prediction = st.session_state["prediction_image"]
    detections = st.session_state["detections"]

    st.divider()
    st.subheader("Prediction Result")

    left, right = st.columns(2)

    with left:
        st.image(
            original_image,
            caption="Original Image",
            use_container_width=True,
        )

    with right:
        st.image(
            prediction,
            caption="Detected Objects",
            use_container_width=True,
        )

    st.subheader("Detection Summary")

    total = len(detections)

    vehicle_count = sum(
        item["Model"] == "Vehicle"
        for item in detections
    )

    traffic_count = sum(
        item["Model"] == "Traffic Light"
        for item in detections
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total detections", total)
    col2.metric("Vehicle detections", vehicle_count)
    col3.metric("Traffic-light detections", traffic_count)

    if detections:
        st.subheader("Detected Objects Table")

        dataframe = pd.DataFrame(detections)

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning(
            "No objects were detected above the "
            "selected confidence threshold."
        )

    # Download prediction image.
    prediction_file = Image.fromarray(prediction)

    import io

    buffer = io.BytesIO()
    prediction_file.save(buffer, format="PNG")

    st.download_button(
        label="⬇️ Download Prediction Image",
        data=buffer.getvalue(),
        file_name="roadvision_prediction.png",
        mime="image/png",
        use_container_width=True,
    )
