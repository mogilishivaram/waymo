from pathlib import Path
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2

ROOT = Path(__file__).resolve().parents[1]
TFRECORD = ROOT / "input" / (
    "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
)
IMAGE_DIR = ROOT / "data" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

frames_seen = 0
images_saved = 0
objects_seen = 0
label_types = {}

dataset = tf.data.TFRecordDataset(TFRECORD, compression_type="")

for record in dataset:
    frame = dataset_pb2.Frame()
    frame.ParseFromString(record.numpy())
    frame_index = frames_seen
    frames_seen += 1

    for image in frame.images:
        if image.name != dataset_pb2.CameraName.FRONT:
            continue

        filename = IMAGE_DIR / f"front_{frame_index:05d}.jpg"
        if not filename.exists():
            with open(filename, "wb") as f:
                f.write(image.image)
            images_saved += 1

    for camera_labels in frame.camera_labels:
        for obj in camera_labels.labels:
            objects_seen += 1
            name = label_pb2.Label.Type.Name(obj.type)
            label_types[name] = label_types.get(name, 0) + 1

    if frames_seen % 10 == 0:
        print(f"Read {frames_seen} frames; newly saved {images_saved} front images")

print("\nExtraction complete")
print("Frames read:", frames_seen)
print("New front images saved:", images_saved)
print("Camera objects counted:", objects_seen)
print("Camera label types:")
for name, count in sorted(label_types.items()):
    print(f"  {name}: {count}")
print("Image folder:", IMAGE_DIR)
