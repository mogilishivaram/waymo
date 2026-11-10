
from pathlib import Path
import random
import cv2
import numpy as np

ROOT = Path("datasets/traffic_lights/coco-s2tld")
OUTPUT = ROOT / "class_mapping_detail_review.jpg"

SAMPLES_PER_CLASS = 8
TARGET_CLASSES = [2, 4]
random.seed(42)


def find_image(stem):
    for split in ("train", "val", "test"):
        path = ROOT / "images" / split / f"{stem}.jpg"
        if path.exists():
            return path
    return None


def make_tile(image, box, class_id, label_path):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box

    # Full scene with the target annotation highlighted.
    context = image.copy()
    cv2.rectangle(context, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.putText(
        context, f"CLASS {class_id}",
        (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 0), 2
    )

    context_h, context_w = 300, 400
    context = cv2.resize(context, (context_w, context_h))

    # Enlarged target crop with padding around the annotated box.
    pad_x = max(20, int((x2 - x1) * 1.5))
    pad_y = max(20, int((y2 - y1) * 1.5))

    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(w, x2 + pad_x)
    bottom = min(h, y2 + pad_y)

    crop = image[top:bottom, left:right].copy()

    if crop.size:
        crop = cv2.resize(crop, (400, 300))
    else:
        crop = np.zeros((300, 400, 3), dtype=np.uint8)

    cv2.putText(
        crop, f"ID {class_id} enlarged",
        (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 0), 2
    )

    tile = np.hstack([context, crop])
    cv2.putText(
        tile, label_path.name,
        (8, 292), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 1
    )
    return tile


tiles = []

for class_id in TARGET_CLASSES:
    candidates = []

    for label_path in (ROOT / "labels").rglob("*.txt"):
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            parts = line.split()
            if len(parts) != 5:
                continue
            if parts[0] != str(class_id):
                continue

            image_path = find_image(label_path.stem)
            if image_path:
                candidates.append((label_path, line_number, line, image_path))

    random.shuffle(candidates)
    selected = 0

    for label_path, line_number, line, image_path in candidates:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        _, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))

        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w - 1, int((cx + bw / 2) * w))
        y2 = min(h - 1, int((cy + bh / 2) * h))

        if x2 <= x1 or y2 <= y1:
            continue

        tiles.append(make_tile(
            image, (x1, y1, x2, y2), class_id, label_path
        ))
        selected += 1

        print(
            f"ID {class_id}: {image_path.name}, "
            f"label line {line_number}, box={(x1, y1, x2, y2)}"
        )

        if selected >= SAMPLES_PER_CLASS:
            break

    print(f"Selected {selected} examples for ID {class_id}")

if not tiles:
    raise RuntimeError("No usable examples found for classes 2 and 4.")

sheet = np.vstack(tiles)
if not cv2.imwrite(str(OUTPUT), sheet):
    raise RuntimeError(f"Could not save review image: {OUTPUT}")

print(f"\nReview image saved to: {OUTPUT.resolve()}")
