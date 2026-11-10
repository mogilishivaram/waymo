
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

OUTPUT = ROOT / "outputs" / "lane_detection" / (
    "map_projection_road_lines_only.png"
)

REPORT = ROOT / "outputs" / "lane_detection" / (
    "map_projection_road_lines_only_report.txt"
)

sys.path.insert(0, str(PROTO_SRC))
from waymo_open_dataset import dataset_pb2


def project_points(points, global_to_camera, intrinsic):
    """Project global points using Waymo camera coordinates.

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

    visible = camera_points[valid]
    x = -visible[:, 1] / visible[:, 0]
    y = -visible[:, 2] / visible[:, 0]

    fx, fy, cx, cy, k1, k2, p1, p2, k3 = intrinsic

    # Brown-Conrady distortion.
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3

    distortion_ok = (
        np.isfinite(radial)
        & (radial >= 0.8)
        & (radial <= 1.2)
    )

    xd = (
        x * radial
        + 2.0 * p1 * x * y
        + p2 * (r2 + 2.0 * x**2)
    )
    yd = (
        y * radial
        + p1 * (r2 + 2.0 * y**2)
        + 2.0 * p2 * x * y
    )

    u = fx * xd + cx
    v = fy * yd + cy

    valid_indices = np.flatnonzero(valid)
    good_indices = valid_indices[distortion_ok]

    pixels[good_indices] = np.column_stack(
        [u[distortion_ok], v[distortion_ok]]
    )

    valid[valid_indices[~distortion_ok]] = False
    valid &= np.isfinite(pixels).all(axis=1)

    return pixels, valid


def get_road_line_type(feature):
    try:
        enum_type = (
            feature.road_line.DESCRIPTOR
            .fields_by_name["type"]
            .enum_type
        )
        return enum_type.values_by_number[
            feature.road_line.type
        ].name
    except (KeyError, AttributeError):
        return str(feature.road_line.type)


def main():
    if not RECORD.exists():
        raise FileNotFoundError(
            f"Waymo TFRecord not found:\n{RECORD}"
        )

    print("Reading the first frame...")

    dataset = tf.data.TFRecordDataset([str(RECORD)])
    serialized = next(iter(dataset.take(1)))

    frame = dataset_pb2.Frame()
    frame.ParseFromString(serialized.numpy())

    camera = next(
        (im for im in frame.images if im.name == 1),
        None,
    )
    calibration = next(
        (cal for cal in frame.context.camera_calibrations
         if cal.name == 1),
        None,
    )

    if camera is None or calibration is None:
        raise RuntimeError(
            "Front camera image or calibration was not found."
        )

    image = Image.open(
        io.BytesIO(camera.image)
    ).convert("RGB")

    draw = ImageDraw.Draw(image)
    width, height = image.size

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

    yellow = (255, 220, 0)

    total_road_lines = 0
    visible_road_lines = 0
    valid_vertices_total = 0
    visible_segments_total = 0

    report = [
        "WAYMO ROAD-LINES-ONLY PROJECTION",
        f"Camera: {camera.name}",
        f"Image dimensions: {width} x {height}",
        "Included: road_line features only",
        "Excluded: road_edge and all other feature types",
        "",
    ]

    for feature in frame.map_features:
        kind = feature.WhichOneof("feature_data")

        # Critical filter: never draw road edges.
        if kind != "road_line":
            continue

        polyline = feature.road_line.polyline
        if len(polyline) < 2:
            continue

        total_road_lines += 1

        points = np.asarray(
            [[p.x, p.y, p.z] for p in polyline],
            dtype=np.float64,
        )

        pixels, valid = project_points(
            points,
            global_to_camera,
            intrinsic,
        )

        valid_vertices_total += int(valid.sum())
        visible_segments = []

        for i in range(len(points) - 1):
            if not (valid[i] and valid[i + 1]):
                continue

            x1, y1 = pixels[i]
            x2, y2 = pixels[i + 1]

            if not np.isfinite([x1, y1, x2, y2]).all():
                continue

            # Avoid connecting distant points across projection gaps.
            if np.hypot(x2 - x1, y2 - y1) > 180:
                continue

            # Ignore segments completely outside the image.
            if (
                max(x1, x2) < 0
                or min(x1, x2) >= width
                or max(y1, y2) < 0
                or min(y1, y2) >= height
            ):
                continue

            draw.line(
                [(int(x1), int(y1)), (int(x2), int(y2))],
                fill=yellow,
                width=3,
            )

            visible_segments.append((x1, y1, x2, y2))
            visible_segments_total += 1

        if not visible_segments:
            continue

        visible_road_lines += 1

        # Add the feature ID near the middle of its visible segments.
        x1, y1, x2, y2 = visible_segments[
            len(visible_segments) // 2
        ]

        label_x = int((x1 + x2) / 2)
        label_y = int((y1 + y2) / 2)
        label = f"ID {feature.id}"

        box = draw.textbbox((label_x, label_y), label)
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text((label_x, label_y), label, fill=yellow)

        coords = np.asarray(
            [(x1, y1) for x1, y1, _, _ in visible_segments]
            + [(x2, y2) for _, _, x2, y2 in visible_segments],
            dtype=np.float64,
        )

        bbox = (
            round(float(coords[:, 0].min()), 1),
            round(float(coords[:, 1].min()), 1),
            round(float(coords[:, 0].max()), 1),
            round(float(coords[:, 1].max()), 1),
        )

        report.append(
            f"ID={feature.id} | "
            f"type={get_road_line_type(feature)} | "
            f"vertices={len(points)} | "
            f"valid_vertices={int(valid.sum())} | "
            f"visible_segments={len(visible_segments)} | "
            f"bbox={bbox}"
        )

    # Draw a small legend.
    draw.rectangle((10, 10, 330, 57), fill=(0, 0, 0))
    draw.text((20, 18), "Yellow: road-line features", fill=yellow)
    draw.text((20, 37), "Road edges excluded", fill=(255, 255, 255))

    # Use fresh, explicit output paths.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("\nSaving image to:", repr(str(OUTPUT)))
    image.save(OUTPUT, format="PNG")

    report.extend([
        "",
        "--- SUMMARY ---",
        f"Road-line features processed: {total_road_lines}",
        f"Features with visible segments: {visible_road_lines}",
        f"Valid projected vertices: {valid_vertices_total}",
        f"Visible segments drawn: {visible_segments_total}",
    ])

    REPORT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nProjection completed.")
    print("Road-line features processed:", total_road_lines)
    print("Features with visible segments:", visible_road_lines)
    print("Visible segments drawn:", visible_segments_total)
    print("\nPreview:", OUTPUT)
    print("Report:", REPORT)
    print("\nImage exists:", OUTPUT.is_file())
    print("Report exists:", REPORT.is_file())


if __name__ == "__main__":
    main()
