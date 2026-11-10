
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[3]
PROTO_SRC = ROOT / "tools" / "waymo-open-dataset" / "src"
RECORD = (
    ROOT / "input" /
    "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
)

sys.path.insert(0, str(PROTO_SRC))
from waymo_open_dataset import dataset_pb2


def points_from_feature(feature):
    feature_type = feature.WhichOneof("feature_data")

    if feature_type not in ("road_line", "road_edge"):
        return None, feature_type

    data = getattr(feature, feature_type)
    points = np.array(
        [[p.x, p.y, p.z] for p in data.polyline],
        dtype=np.float64,
    )

    return points, feature_type


def main():
    if not RECORD.exists():
        raise FileNotFoundError(f"TFRecord not found: {RECORD}")

    dataset = tf.data.TFRecordDataset([str(RECORD)])
    serialized = next(iter(dataset.take(1)))

    frame = dataset_pb2.Frame()
    frame.ParseFromString(serialized.numpy())

    camera = next(
        (im for im in frame.images if im.name == 1), None
    )
    calibration = next(
        (cal for cal in frame.context.camera_calibrations
         if cal.name == 1), None
    )

    if camera is None or calibration is None:
        raise RuntimeError("Front camera or calibration not found.")

    pose = np.asarray(camera.pose.transform).reshape(4, 4)
    extrinsic = np.asarray(
        calibration.extrinsic.transform
    ).reshape(4, 4)

    print("\n--- CAMERA ---")
    print("Image dimensions:", calibration.width, calibration.height)
    print("Camera pose timestamp:", camera.pose_timestamp)
    print("Frame timestamp:", frame.timestamp_micros / 1_000_000)
    print("Camera pose translation:", pose[:3, 3])
    print("Extrinsic translation:", extrinsic[:3, 3])
    print("Intrinsics:", list(calibration.intrinsic))

    print("\nCamera pose matrix:\n", pose)
    print("\nCamera extrinsic matrix:\n", extrinsic)

    transforms = {
        "inverse_extrinsic_inverse_pose":
            np.linalg.inv(extrinsic) @ np.linalg.inv(pose),
        "inverse_pose_inverse_extrinsic":
            np.linalg.inv(pose) @ np.linalg.inv(extrinsic),
        "pose_then_extrinsic":
            extrinsic @ np.linalg.inv(pose),
        "pose_inverse_extrinsic":
            pose @ np.linalg.inv(extrinsic),
    }

    all_points = []
    point_types = []

    for feature in frame.map_features:
        points, feature_type = points_from_feature(feature)
        if points is not None and len(points):
            all_points.append(points)
            point_types.extend([feature_type] * len(points))

    if not all_points:
        print("No road-line or road-edge points found.")
        return

    points = np.concatenate(all_points, axis=0)
    point_types = np.asarray(point_types)

    print("\n--- MAP POINTS ---")
    print("Total road-line/road-edge vertices:", len(points))
    print("Map coordinate min:", points.min(axis=0))
    print("Map coordinate max:", points.max(axis=0))
    print("Map pose offset:", (
        frame.map_pose_offset.x,
        frame.map_pose_offset.y,
        frame.map_pose_offset.z,
    ) if frame.HasField("map_pose_offset") else "not set")

    # Homogeneous coordinates for all map vertices.
    points_h = np.column_stack(
        [points, np.ones(len(points), dtype=np.float64)]
    )

    fx, fy, cx, cy = calibration.intrinsic[:4]

    for name, transform in transforms.items():
        camera_points = (transform @ points_h.T).T[:, :3]

        print(f"\n--- {name} ---")
        print("Camera XYZ min:", camera_points.min(axis=0))
        print("Camera XYZ max:", camera_points.max(axis=0))

        # Test both common depth-axis assumptions.
        for depth_axis in (0, 1, 2):
            depth = camera_points[:, depth_axis]
            positive = depth > 0.1

            print(
                f"Depth axis {depth_axis}: "
                f"{int(positive.sum())}/{len(depth)} vertices "
                "have positive depth"
            )

            if positive.sum() == 0:
                continue

            remaining = camera_points[positive]

            # Print ranges only. This intentionally does not apply
            # image bounds or distortion yet.
            other_axes = [a for a in range(3) if a != depth_axis]
            nx = remaining[:, other_axes[0]] / depth[positive]
            ny = remaining[:, other_axes[1]] / depth[positive]

            u = fx * nx + cx
            v = fy * ny + cy

            inside = (
                (u >= 0) & (u < calibration.width) &
                (v >= 0) & (v < calibration.height)
            )

            print(
                f"  Basic pinhole in-image count: "
                f"{int(inside.sum())}/{len(u)}"
            )
            print(
                f"  Pixel range: u=[{u.min():.1f}, {u.max():.1f}], "
                f"v=[{v.min():.1f}, {v.max():.1f}]"
            )

    print("\nDiagnostic complete. No model weights or project files changed.")


if __name__ == "__main__":
    main()
