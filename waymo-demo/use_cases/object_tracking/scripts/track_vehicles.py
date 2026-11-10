from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = ROOT / "data" / "images"
OUTPUT_DIR = ROOT / "use_cases" / "object_tracking" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_PATH = ROOT / "outputs" / "object_tracking_demo.mp4"

# YOLOv8 + ByteTrack, as specified in our project plan.
model = YOLO(str(ROOT / "use_cases" / "object_tracking" / "models" / "yolov8n.pt"))

images = sorted(IMAGE_DIR.glob("front_*.jpg"))
if not images:
    raise FileNotFoundError(f"No front-camera images found in {IMAGE_DIR}")

writer = None
processed = 0

try:
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        # COCO classes: car=2, motorcycle=3, bus=5, truck=7.
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[2, 3, 5, 7],
            conf=0.25,
            imgsz=640,
            device=0,
            verbose=False,
        )

        annotated = results[0].plot()

        if writer is None:
            height, width = annotated.shape[:2]
            writer = cv2.VideoWriter(
                str(VIDEO_PATH),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create video: {VIDEO_PATH}")

        writer.write(annotated)
        processed += 1

        if processed % 20 == 0 or processed == len(images):
            print(f"Processed {processed}/{len(images)} frames")

finally:
    if writer is not None:
        writer.release()

if processed == 0:
    raise RuntimeError("No frames were processed.")

print(f"Tracking video saved to: {VIDEO_PATH}")
print(f"Frames written: {processed}")
print(f"Approximate video duration: {processed / 10:.1f} seconds")
