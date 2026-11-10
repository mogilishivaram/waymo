
from pathlib import Path

import cv2
from ultralytics import YOLO


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

IMAGE_DIR = ROOT / "data" / "images"

VEHICLE_MODEL_PATH = (
    ROOT / "use_cases" / "object_tracking" / "models" / "yolov8n.pt"
)

TRAFFIC_MODEL_PATH = (
    ROOT / "runs" / "detect" / "runs" / "detect"
    / "traffic_light_detection_v2" / "weights" / "best.pt"
)

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_PATH = OUTPUT_DIR / "combined_scene_demo.mp4"

# Inference settings
IMAGE_SIZE = 640
CONFIDENCE = 0.25
FPS = 10.0
DEVICE = 0

# COCO class IDs:
# car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]

# OpenCV uses BGR colors, not RGB.
TRAFFIC_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 200, 0),
    "off": (160, 160, 160),
    "wait_on": (255, 180, 0),
}


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_class_name(names, class_id):
    """Get a class name from either a list or dictionary."""
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])

    return str(class_id)


def draw_traffic_detections(frame, result, names):
    """Draw traffic-light boxes over the vehicle-annotated frame."""
    if result.boxes is None:
        return frame

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        label = get_class_name(names, class_id)
        color = TRAFFIC_COLORS.get(label.lower(), (255, 180, 0))

        # Traffic-light bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        # Label background
        text = f"TL: {label} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        label_y = max(y1, text_height + 8)

        cv2.rectangle(
            frame,
            (x1, label_y - text_height - 6),
            (x1 + text_width + 6, label_y + baseline - 2),
            color,
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x1 + 3, label_y - 3),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return frame


# --------------------------------------------------
# CHECK REQUIRED FILES
# --------------------------------------------------

if not VEHICLE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Vehicle model not found:\n{VEHICLE_MODEL_PATH}"
    )

if not TRAFFIC_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Traffic-light model not found:\n{TRAFFIC_MODEL_PATH}"
    )

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Image directory not found:\n{IMAGE_DIR}"
    )

images = sorted(IMAGE_DIR.glob("front_*.jpg"))

if not images:
    raise FileNotFoundError(
        f"No front-camera images found in:\n{IMAGE_DIR}"
    )


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

print("Loading vehicle tracking model...")
vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))

print("Loading trained traffic-light model...")
traffic_model = YOLO(str(TRAFFIC_MODEL_PATH))

print(f"Input frames: {len(images)}")
print(f"Output video: {VIDEO_PATH}")


# --------------------------------------------------
# PROCESS FRAMES
# --------------------------------------------------

writer = None
processed = 0

try:
    for image_path in images:
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        # 1. Track vehicles with ByteTrack.
        vehicle_results = vehicle_model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

        annotated = vehicle_results[0].plot()

        # 2. Detect traffic lights in the same frame.
        traffic_results = traffic_model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

        # 3. Overlay traffic-light detections on the vehicle output.
        annotated = draw_traffic_detections(
            annotated,
            traffic_results[0],
            traffic_model.names,
        )

        # 4. Create the video writer using the first valid frame.
        if writer is None:
            height, width = annotated.shape[:2]

            writer = cv2.VideoWriter(
                str(VIDEO_PATH),
                cv2.VideoWriter_fourcc(*"mp4v"),
                FPS,
                (width, height),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Could not create video:\n{VIDEO_PATH}"
                )

        writer.write(annotated)
        processed += 1

        if processed % 10 == 0 or processed == len(images):
            print(
                f"Processed {processed}/{len(images)} frames"
            )

finally:
    if writer is not None:
        writer.release()


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

if processed == 0:
    raise RuntimeError("No frames were processed.")

print("\nCombined inference completed.")
print(f"Video saved to: {VIDEO_PATH}")
print(f"Frames processed: {processed}")
print(f"Approximate duration: {processed / FPS:.1f} seconds")
