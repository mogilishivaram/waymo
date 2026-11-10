import sys
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "waymo-open-dataset" / "src"))

from waymo_open_dataset import dataset_pb2

record = ROOT / "input" / "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
serialized = next(iter(tf.data.TFRecordDataset([str(record)]).take(1)))

frame = dataset_pb2.Frame()
frame.ParseFromString(serialized.numpy())

shown = 0
for feature in frame.map_features:
    kind = feature.WhichOneof("feature_data")
    if kind not in ("lane", "road_line", "road_edge"):
        continue

    geometry = getattr(feature, kind)
    print(f"\nID={feature.id}, feature={kind}")

    for name, field in geometry.DESCRIPTOR.fields_by_name.items():
        value = getattr(geometry, name)
        if field.enum_type:
            enum_name = field.enum_type.values_by_number[value].name
            print(f"  {name}: {enum_name} ({value})")
        elif field.message_type:
            print(f"  {name}: {field.message_type.full_name}")
            if name in ("left_boundary", "right_boundary"):
                print("    ", value)
        elif field.label == field.LABEL_REPEATED:
            print(f"  {name}: {len(value)} items")
        else:
            print(f"  {name}: {value}")

    shown += 1
    if shown >= 12:
        break
