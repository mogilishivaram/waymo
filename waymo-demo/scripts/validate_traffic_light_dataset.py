from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1] / "datasets" / "traffic_lights" / "coco-s2tld"
SPLITS = ("train", "val", "test")
EXPECTED_CLASSES = set(range(5))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_IMAGE_COUNTS = {"train": 4045, "val": 1162, "test": 579}


def image_files(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def validate_label_file(path: Path, image_size: tuple[int, int], errors: list[str]) -> int:
    width, height = image_size
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{path}: could not read label file ({exc})")
        return 0

    if not lines:
        errors.append(f"{path}: empty annotation file")
        return 0

    valid_rows = 0
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_number}: expected 5 fields, got {len(parts)}")
            continue

        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric annotation values")
            continue

        if class_id not in EXPECTED_CLASSES:
            errors.append(f"{path}:{line_number}: class ID {class_id} is outside 0-4")

        if any(value < 0.0 or value > 1.0 for value in values):
            errors.append(f"{path}:{line_number}: normalized coordinate outside [0, 1]")

        _, _, box_width, box_height = values
        if box_width <= 0.0 or box_height <= 0.0:
            errors.append(f"{path}:{line_number}: zero or negative box dimensions")

        x_center, y_center, box_width, box_height = values
        x1 = (x_center - box_width / 2.0) * width
        y1 = (y_center - box_height / 2.0) * height
        x2 = (x_center + box_width / 2.0) * width
        y2 = (y_center + box_height / 2.0) * height
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            errors.append(
                f"{path}:{line_number}: box extends outside image dimensions "
                f"{width}x{height}"
            )

        valid_rows += 1

    return valid_rows


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    split_stats: dict[str, dict[str, int]] = {}
    seen_stems: defaultdict[str, list[str]] = defaultdict(list)
    class_counts: defaultdict[int, int] = defaultdict(int)

    if not ROOT.is_dir():
        print(f"ERROR: dataset root does not exist: {ROOT}")
        return 1

    for split in SPLITS:
        image_dir = ROOT / "images" / split
        label_dir = ROOT / "labels" / split
        if not image_dir.is_dir():
            errors.append(f"missing image directory: {image_dir}")
            continue
        if not label_dir.is_dir():
            errors.append(f"missing label directory: {label_dir}")
            continue

        images = image_files(image_dir)
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        split_stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "annotations": 0,
        }
        if len(images) != EXPECTED_IMAGE_COUNTS[split]:
            errors.append(
                f"{split}: expected {EXPECTED_IMAGE_COUNTS[split]} images, "
                f"found {len(images)}"
            )

        for stem in images:
            seen_stems[stem].append(split)
            if stem not in labels:
                errors.append(f"{split}: image has no label file: {images[stem].name}")

        for stem, label_path in labels.items():
            if stem not in images:
                errors.append(f"{split}: label has no image file: {label_path.name}")
                continue

            image = cv2.imread(str(images[stem]))
            if image is None:
                errors.append(f"{split}: unreadable image: {images[stem]}")
                continue

            height, width = image.shape[:2]
            split_stats[split]["annotations"] += validate_label_file(
                label_path, (width, height), errors
            )
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 5 and parts[0].isdigit():
                    class_counts[int(parts[0])] += 1

        for image_path in images.values():
            if image_path.stem in labels:
                continue
            warnings.append(f"{split}: image without annotations: {image_path.name}")

    duplicates = {
        stem: splits for stem, splits in seen_stems.items() if len(splits) > 1
    }
    for stem, splits in sorted(duplicates.items()):
        errors.append(f"duplicate image filename stem across splits: {stem} ({splits})")

    print(f"Dataset: {ROOT}")
    print("Split summary:")
    for split in SPLITS:
        stats = split_stats.get(split, {"images": 0, "labels": 0, "annotations": 0})
        print(
            f"  {split}: images={stats['images']}, labels={stats['labels']}, "
            f"annotations={stats['annotations']}"
        )
    print(f"Duplicate image stems: {len(duplicates)}")
    print(
        "Class counts: "
        + ", ".join(
            f"{class_id}={class_counts[class_id]}" for class_id in sorted(EXPECTED_CLASSES)
        )
    )
    print(f"Warnings: {len(warnings)}")
    print(f"Errors: {len(errors)}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings[:20]:
            print(f"  - {warning}")
        if len(warnings) > 20:
            print(f"  ... {len(warnings) - 20} more warning(s)")

    if errors:
        print("\nErrors:")
        for error in errors[:50]:
            print(f"  - {error}")
        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} more error(s)")
        return 1

    print("\nValidation passed: no critical dataset integrity errors found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
