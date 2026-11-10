from pathlib import Path
from collections import Counter
import tensorflow as tf
from waymo_open_dataset import dataset_pb2, label_pb2

path = Path(__file__).resolve().parents[1] / "input" / (
    "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
)

camera_type_counts = Counter()
type_counts = Counter()
frames = 0

for record in tf.data.TFRecordDataset(path):
    frame = dataset_pb2.Frame()
    frame.ParseFromString(record.numpy())
    frames += 1

    for camera_labels in frame.camera_labels:
        camera_id = int(camera_labels.name)
        for obj in camera_labels.labels:
            label_type = label_pb2.Label.Type.Name(obj.type)
            type_counts[label_type] += 1
            camera_type_counts[(camera_id, label_type)] += 1

print("Frames:", frames)
print("\nAll camera label types:")
for name, count in sorted(type_counts.items()):
    print(f"{name}: {count}")

print("\nBreakdown by camera ID (Waymo: FRONT=1, FRONT_LEFT=2, FRONT_RIGHT=3, SIDE_LEFT=4, SIDE_RIGHT=5):")
for (camera_id, label_type), count in sorted(camera_type_counts.items()):
    print(f"Camera {camera_id} | {label_type}: {count}")
