
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "waymo-open-dataset" / "src"))

from waymo_open_dataset import dataset_pb2

RECORD = ROOT / "input" / (
    "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
)
OUTPUT = ROOT / "outputs" / "lane_detection" / "front_projected_lidar_labels.png"


def main():
    dataset = tf.data.TFRecordDataset([str(RECORD)])
    raw = next(iter(dataset.take(1)))

    frame = dataset_pb2.Frame()
    frame.ParseFromString(raw.numpy())

    camera = next(image for image in frame.images if image.name == 1)
    image = cv2.imdecode(
        np.frombuffer(camera.image, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise RuntimeError("Could not decode front camera image.")

    overlay = image.copy()
    camera_labels = [
        labels for labels in frame.projected_lidar_labels
        if labels.name == 1
    ]

    count = 0
    for camera_label_group in camera_labels:
        for label in camera_label_group.labels:
            box = label.box
            x1 = int(round(box.center_x - box.length / 2))
            y1 = int(round(box.center_y - box.width / 2))
            x2 = int(round(box.center_x + box.length / 2))
            y2 = int(round(box.center_y + box.width / 2))

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                overlay,
                f"ID {label.id}",
                (max(0, x1), max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            count += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), overlay):
        raise RuntimeError(f"Could not save image: {OUTPUT}")

    print("Front-camera projected LiDAR labels:", count)
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
