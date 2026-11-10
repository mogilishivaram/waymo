
import sys
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[3]
PROTO_SRC = ROOT / "tools" / "waymo-open-dataset" / "src"
RECORD = ROOT / "input" / (
    "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
)

sys.path.insert(0, str(PROTO_SRC))
from waymo_open_dataset import dataset_pb2


def main():
    dataset = tf.data.TFRecordDataset([str(RECORD)])
    raw = next(iter(dataset.take(1)))

    frame = dataset_pb2.Frame()
    frame.ParseFromString(raw.numpy())

    print("Frame timestamp:", frame.timestamp_micros / 1e6)
    print("Frame pose present:", frame.HasField("pose"))

    for camera in frame.images:
        if camera.name != 1:
            continue

        print("\nFRONT CAMERA TIMING")
        print("Pose timestamp:", camera.pose_timestamp)
        print("Frame minus pose timestamp:",
              frame.timestamp_micros / 1e6 - camera.pose_timestamp)
        print("Shutter duration:", camera.shutter)
        print("Camera trigger time:", camera.camera_trigger_time)
        print("Camera readout done time:", camera.camera_readout_done_time)
        print("Velocity:",
              camera.velocity.v_x,
              camera.velocity.v_y,
              camera.velocity.v_z)
        print("Angular velocity:",
              camera.velocity.w_x,
              camera.velocity.w_y,
              camera.velocity.w_z)

    print("\nROAD-LINE FEATURE TYPES")
    for feature in frame.map_features:
        if feature.WhichOneof("feature_data") != "road_line":
            continue

        line = feature.road_line
        enum_type = line.DESCRIPTOR.fields_by_name["type"].enum_type
        type_name = enum_type.values_by_number[line.type].name

        print(
            f"ID={feature.id}, type={type_name}, "
            f"vertices={len(line.polyline)}"
        )


if __name__ == "__main__":
    main()
