
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[3]
PROTO_SRC = ROOT / "tools" / "waymo-open-dataset" / "src"
RECORD = ROOT / "input" / (
    "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
)
OUTPUT_DIR = ROOT / "outputs" / "lane_detection"
OUTPUT_IMAGE = OUTPUT_DIR / "map_projection_rolling_shutter.png"
OUTPUT_REPORT = OUTPUT_DIR / "map_projection_rolling_shutter_report.txt"

CAMERA_NAME = 1  # FRONT
MAX_ITERATIONS = 4
TIME_TOLERANCE = 1e-5


def skew(v):
    """Return the 3x3 cross-product matrix of a vector."""
    x, y, z = v
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)


def transform_matrix(transform_proto):
    """Convert a Waymo row-major 4x4 transform to NumPy."""
    return np.asarray(
        transform_proto.transform, dtype=np.float64
    ).reshape(4, 4)


def undistort_normalized(u_distorted, v_distorted, intrinsics):
    """Invert Waymo's Brown-Conrady distortion iteratively."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = intrinsics

    u_target = (u_distorted - cx) / fx
    v_target = (v_distorted - cy) / fy

    u = u_target
    v = v_target

    for _ in range(20):
        r2 = u * u + v * v
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

        if abs(radial) < 1e-12:
            break

        tangential_u = 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u)
        tangential_v = 2.0 * p2 * u * v + p1 * (r2 + 2.0 * v * v)

        new_u = (u_target - tangential_u) / radial
        new_v = (v_target - tangential_v) / radial

        if (new_u - u) ** 2 + (new_v - v) ** 2 < 1e-16:
            u, v = new_u, new_v
            break

        u, v = new_u, new_v

    return u, v


def distort_and_project(u, v, intrinsics, width, height):
    """Project normalized camera coordinates to distorted pixels."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = intrinsics

    r2 = u * u + v * v
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    if not 0.8 <= radial <= 1.2:
        return None

    ud = u * radial + 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u)
    vd = v * radial + p1 * (r2 + 2.0 * v * v) + 2.0 * p2 * u * v

    pixel_u = ud * fx + cx
    pixel_v = vd * fy + cy

    if not np.isfinite(pixel_u) or not np.isfinite(pixel_v):
        return None

    return float(pixel_u), float(pixel_v)


class RollingShutterProjector:
    """NumPy approximation of Waymo's WorldToImage camera model."""

    def __init__(self, calibration, camera_image):
        self.intrinsics = np.asarray(
            calibration.intrinsic, dtype=np.float64
        )
        self.width = int(calibration.width)
        self.height = int(calibration.height)
        self.direction = int(calibration.rolling_shutter_direction)

        self.pose_timestamp = float(camera_image.pose_timestamp)
        self.trigger_time = float(camera_image.camera_trigger_time)
        self.readout_done_time = float(camera_image.camera_readout_done_time)
        self.shutter = float(camera_image.shutter)

        self.vehicle_to_global = transform_matrix(camera_image.pose)
        self.camera_to_vehicle = transform_matrix(calibration.extrinsic)

        self.camera_to_global = (
            self.vehicle_to_global @ self.camera_to_vehicle
        )

        self.rotation_vehicle_global = self.vehicle_to_global[:3, :3]
        self.rotation_camera_vehicle = self.camera_to_vehicle[:3, :3]
        self.rotation_camera_global = self.camera_to_global[:3, :3]
        self.camera_position_global = self.camera_to_global[:3, 3]

        velocity = camera_image.velocity
        self.linear_velocity_global = np.array(
            [velocity.v_x, velocity.v_y, velocity.v_z],
            dtype=np.float64,
        )
        self.angular_velocity_vehicle = np.array(
            [velocity.w_x, velocity.w_y, velocity.w_z],
            dtype=np.float64,
        )

        angular_velocity_global = (
            self.rotation_vehicle_global @ self.angular_velocity_vehicle
        )

        # Camera motion includes the camera's offset from the vehicle origin.
        camera_offset_global = (
            self.rotation_vehicle_global
            @ self.camera_to_vehicle[:3, 3]
        )

        self.camera_linear_velocity_global = (
            self.linear_velocity_global
            + np.cross(angular_velocity_global, camera_offset_global)
        )

        angular_velocity_camera = (
            self.rotation_camera_vehicle.T
            @ self.angular_velocity_vehicle
        )

        self.skew_omega = skew(angular_velocity_camera)
        self.camera_global_to_camera_at_pose = (
            self.rotation_camera_global.T
        )
        self.skew_omega_dcm = (
            -self.skew_omega @ self.camera_global_to_camera_at_pose
        )

        self.readout_duration = (
            self.readout_done_time - self.trigger_time - self.shutter
        )

        if self.readout_duration < 0:
            raise ValueError(
                "Invalid camera timing: readout duration is negative."
            )

        fx, fy, cx, cy = self.intrinsics[:4]
        if fx <= 0 or fy <= 0:
            raise ValueError("Invalid camera focal length.")

        # Compute the timestamp of the principal-point pixel.
        self.principal_u = float(cx)
        self.principal_v = float(cy)

        if self.direction == 2:  # LEFT_TO_RIGHT
            offset_fraction = cx / self.width
        elif self.direction == 4:  # RIGHT_TO_LEFT
            offset_fraction = (self.width - cx) / self.width
        elif self.direction == 1:  # TOP_TO_BOTTOM
            offset_fraction = cy / self.height
        elif self.direction == 3:  # BOTTOM_TO_TOP
            offset_fraction = (self.height - cy) / self.height
        elif self.direction == 5:  # GLOBAL_SHUTTER
            offset_fraction = 0.0
        else:
            raise ValueError(
                f"Unsupported rolling-shutter direction: {self.direction}"
            )

        principal_time = (
            self.trigger_time
            + 0.5 * self.shutter
            + self.readout_duration * offset_fraction
        )
        self.pose_time_offset = self.pose_timestamp - principal_time

        self.horizontal_readout = self.direction in (2, 4)

        if self.direction == 5:
            self.readout_time_factor = 0.0
        elif self.horizontal_readout:
            u_left, _ = undistort_normalized(
                0.0, 0.5 * self.height, self.intrinsics
            )
            u_right, _ = undistort_normalized(
                float(self.width), 0.5 * self.height, self.intrinsics
            )
            normalized_range = u_right - u_left
            sign = -1.0 if self.direction == 4 else 1.0
            self.readout_time_factor = (
                sign * self.readout_duration / normalized_range
            )
        else:
            _, v_top = undistort_normalized(
                0.5 * self.width, 0.0, self.intrinsics
            )
            _, v_bottom = undistort_normalized(
                0.5 * self.width, float(self.height), self.intrinsics
            )
            normalized_range = v_bottom - v_top
            sign = -1.0 if self.direction == 3 else 1.0
            self.readout_time_factor = (
                sign * self.readout_duration / normalized_range
            )

    def project(self, point_global):
        """Project a static map point, iterating for its rolling-shutter time."""
        point_global = np.asarray(point_global, dtype=np.float64)
        t_h = 0.0
        residual = float("inf")
        normalized = None
        depth = -1.0

        for _ in range(MAX_ITERATIONS):
            camera_global_to_camera = (
                self.camera_global_to_camera_at_pose
                + t_h * self.skew_omega_dcm
            )

            camera_position = (
                self.camera_position_global
                + t_h * self.camera_linear_velocity_global
            )

            point_camera = camera_global_to_camera @ (
                point_global - camera_position
            )

            x, y, z = point_camera
            if not np.all(np.isfinite(point_camera)) or x <= 0.0:
                return None

            u = -y / x
            v = -z / x
            depth = float(x)
            normalized = (u, v)

            spacing = u if self.horizontal_readout else v

            residual = (
                t_h
                - spacing * self.readout_time_factor
                + self.pose_time_offset
            )

            jacobian_landmark = (
                camera_global_to_camera
                @ (-self.camera_linear_velocity_global)
                - self.skew_omega @ point_camera
            )

            jacobian = 1.0
            if self.horizontal_readout:
                jacobian -= (
                    self.readout_time_factor / x
                ) * (u * jacobian_landmark[0] - jacobian_landmark[1])
            elif self.direction != 5:
                jacobian -= (
                    self.readout_time_factor / x
                ) * (v * jacobian_landmark[0] - jacobian_landmark[2])

            if abs(jacobian) < 1e-12:
                return None

            delta_t = -residual / jacobian
            t_h += delta_t

            if abs(residual) <= TIME_TOLERANCE:
                break

        if normalized is None or not np.isfinite(residual):
            return None

        # Recompute the point at the final estimated capture time.
        camera_global_to_camera = (
            self.camera_global_to_camera_at_pose
            + t_h * self.skew_omega_dcm
        )
        camera_position = (
            self.camera_position_global
            + t_h * self.camera_linear_velocity_global
        )
        point_camera = camera_global_to_camera @ (
            point_global - camera_position
        )

        if point_camera[0] <= 0.0:
            return None

        u = -point_camera[1] / point_camera[0]
        v = -point_camera[2] / point_camera[0]

        pixel = distort_and_project(
            u, v, self.intrinsics, self.width, self.height
        )

        if pixel is None:
            return None

        return pixel


def get_feature_type(feature):
    kind = feature.WhichOneof("feature_data")
    if kind != "road_line":
        return None

    line = feature.road_line
    enum_field = line.DESCRIPTOR.fields_by_name["type"]
    enum_value = enum_field.enum_type.values_by_number.get(line.type)
    type_name = enum_value.name if enum_value else str(line.type)
    return type_name, line.polyline


def main():
    if not RECORD.exists():
        raise FileNotFoundError(f"TFRecord not found: {RECORD}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = tf.data.TFRecordDataset([str(RECORD)])
    raw = next(iter(dataset.take(1)))

    from waymo_open_dataset import dataset_pb2

    frame = dataset_pb2.Frame()
    frame.ParseFromString(raw.numpy())

    camera_image = next(
        (image for image in frame.images if image.name == CAMERA_NAME),
        None,
    )
    if camera_image is None:
        raise RuntimeError("Front camera image was not found in the frame.")

    calibration = next(
        (
            camera
            for camera in frame.context.camera_calibrations
            if camera.name == CAMERA_NAME
        ),
        None,
    )
    if calibration is None:
        raise RuntimeError("Front camera calibration was not found.")

    image_array = np.frombuffer(camera_image.image, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not decode the front camera JPEG.")

    projector = RollingShutterProjector(calibration, camera_image)
    overlay = image.copy()

    report = [
        "WAYMO ROLLING-SHUTTER MAP PROJECTION",
        f"Camera: FRONT ({CAMERA_NAME})",
        f"Image dimensions: {image.shape[1]} x {image.shape[0]}",
        f"Rolling-shutter direction enum: {projector.direction}",
        f"Readout duration: {projector.readout_duration:.9f} seconds",
        f"Pose timestamp offset: {projector.pose_time_offset:.9f} seconds",
        "Features: road_line only",
        "",
    ]

    feature_count = 0
    visible_features = 0
    total_vertices = 0
    projected_vertices = 0
    drawn_segments = 0

    for feature in frame.map_features:
        feature_data = get_feature_type(feature)
        if feature_data is None:
            continue

        type_name, polyline = feature_data
        feature_count += 1

        points = [
            (vertex.x, vertex.y, vertex.z)
            for vertex in polyline
        ]
        total_vertices += len(points)

        projected = [projector.project(p) for p in points]
        projected_vertices += sum(p is not None for p in projected)

        feature_segments = 0

        for index in range(len(projected) - 1):
            p1 = projected[index]
            p2 = projected[index + 1]

            if p1 is None or p2 is None:
                continue

            x1, y1 = p1
            x2, y2 = p2

            # Keep only segments whose endpoints are inside the image.
            # This avoids drawing long false lines from extreme off-image
            # projections. Segments crossing the image boundary may be omitted.
            if not (
                0 <= x1 < image.shape[1]
                and 0 <= y1 < image.shape[0]
                and 0 <= x2 < image.shape[1]
                and 0 <= y2 < image.shape[0]
            ):
                continue

            cv2.line(
                overlay,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            drawn_segments += 1
            feature_segments += 1

        if feature_segments:
            visible_features += 1
            valid_points = [p for p in projected if p is not None]
            if valid_points:
                center = np.mean(np.asarray(valid_points), axis=0)
                label_x = int(np.clip(center[0], 5, image.shape[1] - 180))
                label_y = int(np.clip(center[1], 25, image.shape[0] - 5))

                cv2.putText(
                    overlay,
                    f"ID {feature.id}: {type_name}",
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        report.append(
            f"ID={feature.id} | type={type_name} | "
            f"vertices={len(points)} | "
            f"projected={sum(p is not None for p in projected)} | "
            f"drawn_segments={feature_segments}"
        )

    report.extend([
        "",
        "--- SUMMARY ---",
        f"Road-line features processed: {feature_count}",
        f"Features with visible segments: {visible_features}",
        f"Map vertices processed: {total_vertices}",
        f"Successfully projected vertices: {projected_vertices}",
        f"Visible segments drawn: {drawn_segments}",
    ])

    if not cv2.imwrite(str(OUTPUT_IMAGE), overlay):
        raise RuntimeError(f"Could not save image: {OUTPUT_IMAGE}")

    OUTPUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"\nPreview: {OUTPUT_IMAGE}")
    print(f"Report:  {OUTPUT_REPORT}")


if __name__ == "__main__":
    sys.path.insert(0, str(PROTO_SRC))
    main()
