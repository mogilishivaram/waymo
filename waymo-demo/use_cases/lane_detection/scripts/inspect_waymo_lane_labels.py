import sys
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "waymo-open-dataset" / "src"))

from waymo_open_dataset import dataset_pb2

record = ROOT / "input" / "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"

dataset = tf.data.TFRecordDataset([str(record)])

for i, serialized in enumerate(dataset.take(3), start=1):
    frame = dataset_pb2.Frame()
    frame.ParseFromString(serialized.numpy())

    print(f"\nFRAME {i}")
    print("3D laser labels:", len(frame.laser_labels))
    print("Camera 2D labels:", len(frame.camera_labels))
    print("Projected lidar labels:", len(frame.projected_lidar_labels))
    print("Map features:", len(frame.map_features))

    for camera in frame.images:
        seg = camera.camera_segmentation_label
        print(
            f"Camera {camera.name}: "
            f"image={len(camera.image)} bytes, "
            f"segmentation={bool(seg.ListFields())}"
        )
