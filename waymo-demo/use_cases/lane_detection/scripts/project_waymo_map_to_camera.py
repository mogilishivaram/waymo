
import sys
import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
PROTO_SRC = ROOT / "tools" / "waymo-open-dataset" / "src"

RECORD = ROOT / "input" / (
    "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
print("Saving preview to:", repr(str(OUTPUT)))

OUTPUT = ROOT / "outputs" / "lane_detection" / (
    "map_projection_road_lines_only_v2.png"
)

REPORT = ROOT / "outputs" / "lane_detection" / (
    "map_projection_road_lines_only_v2_report.txt"
)

sys.path.insert(0, str(PROTO_SRC))
from waymo_open_dataset import dataset_pb2


def project_points(points, global_to_camera, intrinsic):
    """Project global map points using Waymo camera coordinates.

    Waymo camera axes:
        +X = forward
        +Y = left
        +Z = up
    """
    points = np.asarray(points, dtype=np.float64)

    points_h = np.column_stack(
        [points, np.ones(len(points), dtype=np.float64)]
    )
    camera_points = (global_to_camera @ points_h.T).T[:, :3]

    depth = camera_points[:, 0]
    valid = depth > 0.1
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)

    if not valid.any():
        return pixels, valid

    visible_points = camera_points[valid]

    # Normalize according to Waymo's camera convention.
    x = -visible_points[:, 1] / visible_points[:, 0]
    y = -visible_points[:, 2] / visible_points[:, 0]

    fx, fy, cx, cy, k1, k2, p1, p2, k3 = intrinsic

    # Brown-Conrady lens distortion.
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3

    distortion_ok = (
        np.isfinite(radial)
        & (radial >= 0.8)
        & (radial <= 1.2)
    )

    x_distorted = (
        x * radial
        + 2.0 * p1 * x * y
        + p2 * (r2 + 2.0 * x**2)
    )

    y_distorted = (
        y * radial
        + p1 * (r2 + 2.0 * y**2)
        + 2.0 * p2 * x * y
    )

    u = fx * x_distorted + cx
    v = fy * y_distorted + cy

    valid_indices = np.flatnonzero(valid)
    good_indices = valid_indices[distortion_ok]

    pixels[good_indices, 0] = u[distortion_ok]
    pixels[good_indices, 1] = v[distortion_ok]

    valid[valid_indices[~distortion_ok]] = False

    # Reject non-finite pixel coordinates.
    finite = np.isfinite(pixels).all(axis=1)
    valid &= finite

    return pixels, valid


def get_road_line_type(feature):
    """Return a readable road-line subtype."""
    road_line = feature.road_line

    try:
        enum_descriptor = (
            road_line.DESCRIPTOR.fields_by_name["type"].enum_type
        )
        return enum_descriptor.values_by_number[
            road_line.type
        ].name
    except (KeyError, AttributeError):
        return str(road_line.type)


def main():
    if not RECORD.exists():
        raise FileNotFoundError(
            f"Waymo TFRecord not found:\n{RECORD}"
        )

    print("Reading the first Waymo frame...")

    dataset = tf.data.TFRecordDataset([str(RECORD)])
    serialized = next(iter(dataset.take(1)))

    frame = dataset_pb2.Frame()
    frame.ParseFromString(serialized.numpy())

    camera = next(
        (image for image in frame.images if image.name == 1),
        None,
    )

    calibration = next(
        (
            item
            for item in frame.context.camera_calibrations
            if item.name == 1
        ),
        None,
    )

    if camera is None:
        raise RuntimeError("Front camera image was not found.")

    if calibration is None:
        raise RuntimeError("Front camera calibration was not found.")

    image = Image.open(
        io.BytesIO(camera.image)
    ).convert("RGB")

    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Camera-to-global = vehicle-to-global @ camera-to-vehicle.
    vehicle_to_global = np.asarray(
        camera.pose.transform,
        dtype=np.float64,
    ).reshape(4, 4)

    camera_to_vehicle = np.asarray(
        calibration.extrinsic.transform,
        dtype=np.float64,
    ).reshape(4, 4)

    camera_to_global = vehicle_to_global @ camera_to_vehicle
    global_to_camera = np.linalg.inv(camera_to_global)

    intrinsic = np.asarray(
        calibration.intrinsic,
        dtype=np.float64,
    )

    # Only actual road-line map features are included.
    line_color = (255, 220, 0)
    feature_count = 0
    visible_feature_count = 0
    valid_vertex_count = 0
    visible_segment_count = 0

    report = [
        "WAYMO ROAD-LINE PROJECTION REPORT",
        f"Camera: {camera.name}",
        f"Image dimensions: {width} x {height}",
        "Included features: road_line only",
        "Excluded features: road_edge and all other map features",
        "",
    ]

    for feature in frame.map_features:
        kind = feature.WhichOneof("feature_data")

        if kind != "road_line":
            continue

        polyline = feature.road_line.polyline

        if len(polyline) < 2:
            continue

        feature_count += 1

        points = np.asarray(
            [[point.x, point.y, point.z] for point in polyline],
            dtype=np.float64,
        )

        pixels, valid = project_points(
            points,
            global_to_camera,
            intrinsic,
        )

        valid_vertex_count += int(valid.sum())
        feature_segments = []

        for index in range(len(points) - 1):
            if not (valid[index] and valid[index + 1]):
                continue

            x1, y1 = pixels[index]
            x2, y2 = pixels[index + 1]

            if not np.isfinite([x1, y1, x2, y2]).all():
                continue

            # Avoid connecting points across large projection gaps.
            if np.hypot(x2 - x1, y2 - y1) > 180:
                continue

            # Skip segments completely outside the image.
            if (
                max(x1, x2) < 0
                or min(x1, x2) >= width
                or max(y1, y2) < 0
                or min(y1, y2) >= height
            ):
                continue

            draw.line(
                [(int(x1), int(y1)), (int(x2), int(y2))],
                fill=line_color,
                width=3,
            )

            feature_segments.append(
                (x1, y1, x2, y2)
            )
            visible_segment_count += 1

        if not feature_segments:
            continue

        visible_feature_count += 1

        # Label the feature near the middle of its visible segments.
        x1, y1, x2, y2 = feature_segments[
            len(feature_segments) // 2
        ]

        label_x = int((x1 + x2) / 2)
        label_y = int((y1 + y2) / 2)

        label = f"ID {feature.id}"
        text_box = draw.textbbox(
            (label_x, label_y),
            label,
        )

        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text(
            (label_x, label_y),
            label,
            fill=line_color,
        )

        coords = np.asarray(
            [
                (x1, y1)
                for x1, y1, _, _ in feature_segments
            ]
            + [
                (x2, y2)
                for _, _, x2, y2 in feature_segments
            ],
            dtype=np.float64,
        )

        bbox = (
            round(float(coords[:, 0].min()), 1),
            round(float(coords[:, 1].min()), 1),
            round(float(coords[:, 0].max()), 1),
            round(float(coords[:, 1].max()), 1),
        )

        line_type = get_road_line_type(feature)

        report.append(
            f"ID={feature.id} | "
            f"type={line_type} | "
            f"vertices={len(points)} | "
            f"valid_vertices={int(valid.sum())} | "
            f"visible_segments={len(feature_segments)} | "
            f"bbox={bbox}"
        )

    # Legend.
    draw.rectangle((10, 10, 300, 55), fill=(0, 0, 0))
    draw.text(
        (20, 18),
        "Yellow: Waymo road-line features",
        fill=line_color,
    )
    draw.text(
        (20, 36),
        "Road edges excluded",
        fill=(255, 255, 255),
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    image.save(OUTPUT)

    report.extend([
        "",
        "--- SUMMARY ---",
        f"Road-line features in frame: {feature_count}",
        f"Features with visible segments: {visible_feature_count}",
        f"Valid projected vertices: {valid_vertex_count}",
        f"Visible segments drawn: {visible_segment_count}",
    ])

    REPORT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nProjection completed.")
    print(f"Road-line features in frame: {feature_count}")
    print(f"Features with visible segments: {visible_feature_count}")
    print(f"Valid projected vertices: {valid_vertex_count}")
    print(f"Visible segments drawn: {visible_segment_count}")
    print(f"\nPreview saved:\n{OUTPUT}")
    print(f"\nReport saved:\n{REPORT}")
    print(
        "\nNote: this is a static-pose projection approximation; "
        "rolling-shutter compensation is not applied."
    )


if __name__ == "__main__":
    main()
