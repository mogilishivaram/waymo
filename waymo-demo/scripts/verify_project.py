from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CRITICAL_FILES = {
    "input/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord": "Waymo TFRecord",
    "data/images/front_00000.jpg": "extracted front image",
    "data/yolo_vehicle/data.yaml": "prepared YOLO dataset",
    "datasets/object_tracking/data.yaml": "object-tracking dataset reference",
    "use_cases/object_tracking/models/yolov8n.pt": "YOLOv8 model",
    "use_cases/object_tracking/models/yolo11n.pt": "preserved YOLO11 model",
    "use_cases/object_tracking/models/waymo_vehicle_baseline-3/weights/best.pt": "trained YOLO11 weights",
    "outputs/object_tracking_demo.mp4": "object-tracking demo",
}

PENDING_DIRECTORIES = {
    "datasets/traffic_lights/images": "traffic-light images",
    "datasets/traffic_lights/annotations": "traffic-light annotations",
    "datasets/lanes/images": "lane images",
    "datasets/lanes/annotations": "lane annotations",
}


def main() -> int:
    missing = []
    print(f"Project: {ROOT}")
    for relative, description in CRITICAL_FILES.items():
        path = ROOT / relative
        status = "OK" if path.exists() else "MISSING"
        print(f"[{status}] {description}: {relative}")
        if not path.exists():
            missing.append(relative)

    print("\nPending-use-case directories:")
    for relative, description in PENDING_DIRECTORIES.items():
        path = ROOT / relative
        print(f"[{'PRESENT' if path.exists() else 'MISSING'}] {description}: {relative}")

    if missing:
        print(f"\nVerification failed: {len(missing)} critical asset(s) missing.")
        return 1
    print("\nVerification passed: all critical preserved assets are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
