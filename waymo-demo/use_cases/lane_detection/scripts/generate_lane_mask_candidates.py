
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = ROOT / "datasets" / "lanes"

SOURCE_DIRS = [
    DATASET_ROOT / "images",
    ROOT / "input",
]

GENERATED_DIR_NAMES = {
    "mask_candidates", "mask_yellow", "mask_white",
    "mask_overlays", "mask_preview", "annotations",
    "mask_candidates_v2", "mask_yellow_v2",
    "mask_white_v2", "mask_overlays_v2",
}

# New output folders. Existing outputs are preserved.
OUTPUT_COMBINED = DATASET_ROOT / "mask_candidates_v2"
OUTPUT_YELLOW = DATASET_ROOT / "mask_yellow_v2"
OUTPUT_WHITE = DATASET_ROOT / "mask_white_v2"
OUTPUT_OVERLAYS = DATASET_ROOT / "mask_overlays_v2"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Existing vehicle detector. This script never modifies its weights.
VEHICLE_MODEL_PATH = (
    ROOT / "use_cases" / "object_tracking" / "models"
    / "waymo_vehicle_baseline-3" / "weights" / "best.pt"
)

USE_VEHICLE_SUPPRESSION = True
VEHICLE_CONFIDENCE = 0.30
VEHICLE_IMAGE_SIZE = 640
VEHICLE_BOX_PADDING = 0.04  # fraction of box width/height

# Tune these only after inspecting the v2 overlays.
WHITE_MIN_VALUE = 145
WHITE_MAX_SATURATION = 85
YELLOW_MIN_SATURATION = 65
YELLOW_MIN_VALUE = 85


def find_images():
    """Find original photographs, excluding generated outputs."""
    found = {}

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            continue

        for path in source_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if any(
                part.lower() in GENERATED_DIR_NAMES
                for part in path.parts
            ):
                continue
            if path.stem.endswith(
                ("_mask", "_overlay", "_yellow", "_white")
            ):
                continue

            found[str(path.resolve()).lower()] = path

    return sorted(found.values())


def road_region(height, width):
    """
    Approximate front-camera road ROI.
    It excludes much of the sky and buildings, but cannot
    understand road geometry or follow every curve.
    """
    roi = np.zeros((height, width), dtype=np.uint8)

    polygon = np.array([
        [int(width * 0.10), height - 1],
        [int(width * 0.34), int(height * 0.40)],
        [int(width * 0.67), int(height * 0.40)],
        [int(width * 0.95), height - 1],
    ], dtype=np.int32)

    cv2.fillPoly(roi, [polygon], 255)
    return roi


def load_vehicle_model():
    if not USE_VEHICLE_SUPPRESSION:
        print("Vehicle suppression disabled.")
        return None

    if not VEHICLE_MODEL_PATH.exists():
        print("WARNING: Vehicle weights not found:")
        print(f"  {VEHICLE_MODEL_PATH}")
        print("Continuing without vehicle suppression.")
        return None

    try:
        from ultralytics import YOLO
        model = YOLO(str(VEHICLE_MODEL_PATH))
        print("Loaded existing vehicle detector.")
        return model
    except Exception as exc:
        print("WARNING: Could not load vehicle detector.")
        print(f"Reason: {exc}")
        print("Continuing without vehicle suppression.")
        return None


def vehicle_exclusion_mask(model, image):
    """Return a mask covering detected vehicles and a small margin."""
    height, width = image.shape[:2]
    excluded = np.zeros((height, width), dtype=np.uint8)

    if model is None:
        return excluded

    try:
        results = model.predict(
            source=image,
            conf=VEHICLE_CONFIDENCE,
            imgsz=VEHICLE_IMAGE_SIZE,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return excluded

        boxes = results[0].boxes.xyxy.cpu().numpy()

        for x1, y1, x2, y2 in boxes:
            box_width = max(1, x2 - x1)
            box_height = max(1, y2 - y1)

            pad_x = int(box_width * VEHICLE_BOX_PADDING)
            pad_y = int(box_height * VEHICLE_BOX_PADDING)

            x1 = max(0, int(x1) - pad_x)
            y1 = max(0, int(y1) - pad_y)
            x2 = min(width - 1, int(x2) + pad_x)
            y2 = min(height - 1, int(y2) + pad_y)

            cv2.rectangle(
                excluded, (x1, y1), (x2, y2), 255, thickness=-1
            )

    except Exception as exc:
        print(f"WARNING: Vehicle prediction failed: {exc}")

    return excluded


def remove_small_components(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)

    for component in range(1, count):
        area = stats[component, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == component] = 255

    return cleaned


def make_masks(image, vehicle_model):
    height, width = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # White/light paint: low saturation and relatively high brightness.
    white = cv2.inRange(
        hsv,
        np.array([0, 0, WHITE_MIN_VALUE], dtype=np.uint8),
        np.array([180, WHITE_MAX_SATURATION, 255], dtype=np.uint8),
    )

    # Yellow paint: restrict hue and require stronger saturation.
    yellow = cv2.inRange(
        hsv,
        np.array([15, YELLOW_MIN_SATURATION, YELLOW_MIN_VALUE],
                 dtype=np.uint8),
        np.array([40, 255, 255], dtype=np.uint8),
    )

    # Additional brightness guard for foggy images.
    lightness = lab[:, :, 0]
    bright = cv2.inRange(lightness, 95, 255)

    white = cv2.bitwise_and(white, bright)
    yellow = cv2.bitwise_and(yellow, bright)

    # Apply the road ROI.
    roi = road_region(height, width)
    white = cv2.bitwise_and(white, roi)
    yellow = cv2.bitwise_and(yellow, roi)

    # Remove pixels covered by detected vehicles.
    vehicles = vehicle_exclusion_mask(vehicle_model, image)
    keep = cv2.bitwise_not(vehicles)

    white = cv2.bitwise_and(white, keep)
    yellow = cv2.bitwise_and(yellow, keep)

    # Suppress isolated noise without aggressively thickening markings.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)

    white = remove_small_components(white, min_area=20)
    yellow = remove_small_components(yellow, min_area=6)

    combined = cv2.bitwise_or(white, yellow)

    # Overlay colors: magenta = white candidate, yellow = yellow candidate.
    overlay = image.copy()
    overlay[white > 0] = (255, 0, 255)
    overlay[yellow > 0] = (0, 255, 255)
    overlay = cv2.addWeighted(image, 0.70, overlay, 0.30, 0)

    return combined, yellow, white, overlay


def main():
    for folder in (
        OUTPUT_COMBINED,
        OUTPUT_YELLOW,
        OUTPUT_WHITE,
        OUTPUT_OVERLAYS,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    images = find_images()
    if not images:
        print("ERROR: No source photographs found.")
        print(f"Checked: {SOURCE_DIRS}")
        return

    print(f"Found {len(images)} source images.")
    vehicle_model = load_vehicle_model()

    processed = 0
    failed = 0

    for index, image_path in enumerate(images):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"SKIP unreadable image: {image_path}")
            failed += 1
            continue

        combined, yellow, white, overlay = make_masks(
            image, vehicle_model
        )

        # One output per source image, with stable sequence numbering.
        stem = f"front_{index:05d}"

        outputs = [
            (OUTPUT_COMBINED / f"{stem}_mask.png", combined),
            (OUTPUT_YELLOW / f"{stem}_yellow.png", yellow),
            (OUTPUT_WHITE / f"{stem}_white.png", white),
            (OUTPUT_OVERLAYS / f"{stem}_overlay.jpg", overlay),
        ]

        for output_path, result in outputs:
            if not cv2.imwrite(str(output_path), result):
                raise OSError(f"Could not write output: {output_path}")

        processed += 1

        if processed % 25 == 0 or processed == len(images):
            print(f"Processed {processed}/{len(images)}")

    print("\nV2 candidate generation finished.")
    print(f"Processed: {processed}")
    print(f"Skipped: {failed}")
    print(f"Vehicle suppression: {'ON' if vehicle_model else 'OFF'}")
    print(f"Combined masks: {OUTPUT_COMBINED}")
    print(f"Yellow masks:   {OUTPUT_YELLOW}")
    print(f"White masks:    {OUTPUT_WHITE}")
    print(f"Overlays:       {OUTPUT_OVERLAYS}")
    print("These are still candidate masks, not verified labels.")


if __name__ == "__main__":
    main()
