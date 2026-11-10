from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "input" / (
    "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
)
OUT = ROOT / "data" / "yolo_vehicle"

for split in ("images/train", "images/val", "labels/train", "labels/val"):
    (OUT / split).mkdir(parents=True, exist_ok=True)

samples = []
invalid_boxes = 0
total_boxes = 0

for record in tf.data.TFRecordDataset(RECORD):
    frame = dataset_pb2.Frame()
    frame.ParseFromString(record.numpy())

    image = next(
        (im for im in frame.images
         if im.name == dataset_pb2.CameraName.FRONT),
        None
    )
    if image is None:
        continue

    decoded = cv2.imdecode(
        np.frombuffer(image.image, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if decoded is None:
        print("Skipping an unreadable image")
        continue

    height, width = decoded.shape[:2]
    labels = next(
        (cl for cl in frame.camera_labels
         if cl.name == dataset_pb2.CameraName.FRONT),
        None
    )

    yolo_lines = []
    if labels:
        for obj in labels.labels:
            total_boxes += 1
            box = obj.box

            # Waymo camera box fields: length is vertical,
            # width is horizontal.
            bw, bh = float(box.width), float(box.length)
            cx, cy = float(box.center_x), float(box.center_y)

            if bw <= 0 or bh <= 0:
                invalid_boxes += 1
                continue

            x1 = max(0.0, cx - bw / 2)
            y1 = max(0.0, cy - bh / 2)
            x2 = min(float(width), cx + bw / 2)
            y2 = min(float(height), cy + bh / 2)

            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if bw <= 0 or bh <= 0:
                invalid_boxes += 1
                continue

            yolo_lines.append(
                f"0 {cx/width:.6f} {cy/height:.6f} "
                f"{bw/width:.6f} {bh/height:.6f}"
            )

    samples.append((image.image, yolo_lines))

# Keep chronological order to reduce leakage from adjacent frames.
n = len(samples)
split_index = max(1, int(n * 0.8))

for i, (jpeg, labels) in enumerate(samples):
    split = "train" if i < split_index else "val"
    stem = f"front_{i:05d}"

    with open(OUT / "images" / split / (stem + ".jpg"), "wb") as f:
        f.write(jpeg)

    with open(OUT / "labels" / split / (stem + ".txt"), "w") as f:
        f.write("\n".join(labels))
        if labels:
            f.write("\n")

with open(OUT / "data.yaml", "w") as f:
    f.write(f"path: {OUT}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("names:\n  0: vehicle\n")

print("\nConversion complete")
print("Frames:", n)
print("Training images:", split_index)
print("Validation images:", n - split_index)
print("Front-camera boxes encountered:", total_boxes)
print("Invalid boxes skipped:", invalid_boxes)
print("Dataset directory:", OUT)
