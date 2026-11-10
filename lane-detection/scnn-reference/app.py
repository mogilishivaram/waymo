
import io
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

# --------------------------------------------------
# Project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PREDICT_SCRIPT = BASE_DIR / "predict_green.py"

WEIGHTS_PATH = (
    BASE_DIR
    / "experiments"
    / "vgg_SCNN_DULR_w9"
    / "vgg_SCNN_DULR_w9.pth"
)

# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Lane Detection",
    page_icon="🛣️",
    layout="wide",
)

# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        margin-bottom: 28px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        min-height: 45px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🛣️ AI Lane Detection</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'SCNN-based road lane detection and visualization'
    '</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Project Details")
st.sidebar.write("**Model:** SCNN")
st.sidebar.write("**Task:** Road lane detection")
st.sidebar.write("**Overlay:** Green lane boundaries")
st.sidebar.write("**Inference:** Local Python environment")

st.sidebar.divider()

st.sidebar.info(
    "For better results, upload a front-facing road image "
    "with visible lane markings."
)

st.sidebar.caption(
    "Predictions may be inaccurate on unusual road layouts "
    "or images unlike the model's training data."
)

# --------------------------------------------------
# Verify required files
# --------------------------------------------------
if not PREDICT_SCRIPT.is_file():
    st.error(
        f"Prediction script not found:\n{PREDICT_SCRIPT}"
    )
    st.stop()

if not WEIGHTS_PATH.is_file():
    st.error(
        f"Model checkpoint not found:\n{WEIGHTS_PATH}"
    )
    st.stop()

# --------------------------------------------------
# Image uploader
# --------------------------------------------------
st.subheader("Upload a Road Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG.",
)

if uploaded_file is None:
    st.info(
        "Upload a road image to begin. "
        "You can use demo/demo.jpg from the SCNN repository."
    )
    st.stop()

# --------------------------------------------------
# Read uploaded image
# --------------------------------------------------
try:
    original_image = Image.open(uploaded_file).convert("RGB")
except Exception as exc:
    st.error(f"Could not read the uploaded image: {exc}")
    st.stop()

st.caption(
    f"Image dimensions: {original_image.width} × "
    f"{original_image.height} pixels"
)

st.image(
    original_image,
    caption="Original road image",
    use_container_width=True,
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Detect Lanes", type="primary"):

    with st.spinner(
        "Running SCNN prediction. This may take a moment..."
    ):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                input_path = temp_dir / "input_road.jpg"
                output_path = temp_dir / "lane_result.png"

                # Save uploaded image for the prediction script
                original_image.save(
                    input_path,
                    format="JPEG",
                    quality=95,
                )

                command = [
                    sys.executable,
                    str(PREDICT_SCRIPT),
                    "--img_path",
                    str(input_path),
                    "--weight_path",
                    str(WEIGHTS_PATH),
                    "--output",
                    str(output_path),
                ]

                result = subprocess.run(
                    command,
                    cwd=str(BASE_DIR),
                    capture_output=True,
                    text=True,
                    timeout=180,
                )

                # Handle prediction errors without crashing the app
                if result.returncode != 0:
                    st.session_state["prediction_error"] = (
                        result.stdout,
                        result.stderr,
                    )
                    st.session_state.pop(
                        "prediction_bytes", None
                    )

                elif not output_path.is_file():
                    st.session_state["prediction_error"] = (
                        result.stdout,
                        "Prediction finished, but the output image "
                        "was not created. Check predict_green.py.",
                    )
                    st.session_state.pop(
                        "prediction_bytes", None
                    )

                else:
                    # Read output before temporary files are removed
                    predicted_image = Image.open(
                        output_path
                    ).convert("RGB")

                    image_buffer = io.BytesIO()
                    predicted_image.save(
                        image_buffer,
                        format="PNG",
                    )

                    st.session_state["prediction_bytes"] = (
                        image_buffer.getvalue()
                    )
                    st.session_state["prediction_log"] = (
                        result.stdout
                    )
                    st.session_state.pop(
                        "prediction_error", None
                    )

        except subprocess.TimeoutExpired:
            st.session_state["prediction_error"] = (
                "",
                "Prediction timed out after 180 seconds. "
                "Try a smaller image.",
            )
            st.session_state.pop(
                "prediction_bytes", None
            )

        except Exception as exc:
            st.session_state["prediction_error"] = (
                "",
                str(exc),
            )
            st.session_state.pop(
                "prediction_bytes", None
            )

# --------------------------------------------------
# Display prediction results
# --------------------------------------------------
if "prediction_error" in st.session_state:
    stdout, stderr = st.session_state["prediction_error"]

    st.warning(
        "The model could not produce a usable lane prediction "
        "for this image."
    )

    st.write(
        "Try a clear, front-facing road image with visible lane "
        "markings. This can also happen when the model detects "
        "fewer than two usable boundaries."
    )

    with st.expander("View technical details"):
        if stdout:
            st.text("Standard output:")
            st.code(stdout)

        if stderr:
            st.text("Error output:")
            st.code(stderr)

elif "prediction_bytes" in st.session_state:
    st.success("Lane prediction completed.")

    predicted_bytes = st.session_state["prediction_bytes"]
    predicted_image = Image.open(
        io.BytesIO(predicted_bytes)
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(
            original_image,
            use_container_width=True,
        )

    with col2:
        st.subheader("Lane Detection Result")
        st.image(
            predicted_image,
            use_container_width=True,
        )

    st.download_button(
        label="Download Prediction",
        data=predicted_bytes,
        file_name="lane_detection_result.png",
        mime="image/png",
        type="primary",
    )

    with st.expander("Prediction Details"):
        log = st.session_state.get("prediction_log", "")
        st.text(log or "Prediction completed.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "AI Lane Detection | SCNN | Local Computer Vision Demo"
)
