from pathlib import Path
import tensorflow as tf
from waymo_open_dataset import dataset_pb2

path = Path(__file__).resolve().parents[1] / "input" / (
    "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
)

for record in tf.data.TFRecordDataset(path).take(1):
    frame = dataset_pb2.Frame()
    frame.ParseFromString(record.numpy())

    front_labels = next(
        labels for labels in frame.camera_labels
        if labels.name == dataset_pb2.CameraName.FRONT
    )

    print("Camera label fields:")
    print(front_labels.labels[0].ListFields())

    print("\nBounding box protobuf:")
    print(front_labels.labels[0].box)

    print("\nLabel protobuf descriptor fields:")
    for field in front_labels.labels[0].DESCRIPTOR.fields:
        print(field.name, field.type, field.label)
