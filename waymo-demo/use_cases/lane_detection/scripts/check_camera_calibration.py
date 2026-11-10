
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


def enum_name(field, value):
    enum_type = field.enum_type
    item = enum_type.values_by_number.get(value)
    return item.name if item else f"UNKNOWN_VALUE_{value}"


def main():
    dataset = tf.data.TFRecordDataset([str(RECORD)])
    raw = next(iter(dataset.take(1)))

    frame = dataset_pb2.Frame()
    frame.ParseFromString(raw.numpy())

    camera_image = next(
        image for image in frame.images if image.name == 1
    )

    calibration = next(
        camera
        for camera in frame.context.camera_calibrations
        if camera.name == 1
    )

    center_time = (
        camera_image.camera_trigger_time
        + camera_image.camera_readout_done_time
    ) / 2

    print("FRONT CAMERA CALIBRATION")
    print("Camera name:", dataset_pb2.CameraName.Name.Name(calibration.name))
    print("Image dimensions:", calibration.width, "x", calibration.height)

    direction_field = calibration.DESCRIPTOR.fields_by_name[
        "rolling_shutter_direction"
    ]
    print(
        "Rolling shutter direction:",
        enum_name(direction_field, calibration.rolling_shutter_direction),
    )

    print("\nINTRINSIC PARAMETERS")
    print(list(calibration.intrinsic))

    print("\nCAMERA-TO-VEHICLE EXTRINSIC MATRIX")
    matrix = list(calibration.extrinsic.transform)
    for row in range(4):
        print(matrix[row * 4:(row + 1) * 4])

    print("\nCAMERA TIMING")
    print("Trigger:", camera_image.camera_trigger_time)
    print("Readout end:", camera_image.camera_readout_done_time)
    print("Approximate center exposure time:", center_time)
    print("Pose timestamp:", camera_image.pose_timestamp)
    print(
        "Center exposure minus pose timestamp:",
        center_time - camera_image.pose_timestamp,
        "seconds",
    )


if __name__ == "__main__":
    main()
