from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2

root = Path(__file__).resolve().parents[1]
record_path = root / "input" / (
    "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
)

for record in tf.data.TFRecordDataset(record_path).take(1):
    frame = dataset_pb2.Frame()
    frame.ParseFromString(record.numpy())

    front_image = next(
        image for image in frame.images
        if image.name == dataset_pb2.CameraName.FRONT
    )

    jpeg = np.frombuffer(front_image.image, dtype=np.uint8)
    decoded = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)

    if decoded is None:
        raise RuntimeError("Could not decode the front-camera JPEG")

    height, width = decoded.shape[:2]
    print(f"Front image dimensions: {width} x {height}")

    front_labels = next(
        labels for labels in frame.camera_labels
        if labels.name == dataset_pb2.CameraName.FRONT
    )

    print("Front-camera objects in first frame:", len(front_labels.labels))

    for obj in front_labels.labels[:5]:
        box = obj.box
        print({
            "type": int(obj.type),
            "center_x": box.center_x,
            "center_y": box.center_y,
            "width": box.width,
            "height": box.height,
        })
