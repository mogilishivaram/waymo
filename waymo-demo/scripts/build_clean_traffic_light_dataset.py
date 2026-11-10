
from pathlib import Path
import csv
import shutil

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "datasets" / "traffic_lights" / "coco-s2tld"
REVIEW = ROOT / "datasets" / "traffic_lights" / "processed_review"
MANIFEST = REVIEW / "annotation_review.csv"
PROPOSED = REVIEW / "proposed_labels"
OUTPUT = ROOT / "datasets" / "traffic_lights" / "cleaned"

SPLITS = ("train", "val", "test")
VALID_CLASSES = {0, 1, 2, 3, 4}


def validate_label_file(path):
    """Return an error message, or None when all rows are valid."""
    if not path.exists():
        return "Missing label file"

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            return f"Line {line_number}: expected 5 fields"

        try:
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
        except ValueError:
            return f"Line {line_number}: invalid number"

        if class_id not in VALID_CLASSES:
            return f"Line {line_number}: invalid class ID {class_id}"

        if not all(map(lambda v: 0 <= v <= 1, (x, y, width, height))):
            return f"Line {line_number}: values must be between 0 and 1"

        if width <= 0 or height <= 0:
            return f"Line {line_number}: box width/height must be positive"

        if x - width / 2 < 0 or x + width / 2 > 1:
            return f"Line {line_number}: box exceeds image horizontally"

        if y - height / 2 < 0 or y + height / 2 > 1:
            return f"Line {line_number}: box exceeds image vertically"

    return None


def main():
    if not SOURCE.exists() or not MANIFEST.exists():
        raise FileNotFoundError(
            "Source dataset or annotation_review.csv was not found."
        )

    # Map each reviewed image to its proposed annotation file.
    reviewed = {}
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            split = row["Split"].strip().lower()
            image_name = Path(row["Image"]).name

            if split not in SPLITS:
                raise ValueError(f"Invalid split in manifest: {split}")

            reviewed[(split, image_name)] = row

    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Rename or remove it manually "
            "after checking its contents before running again."
        )

    counts = {
        "copied": 0,
        "reviewed": 0,
        "skipped": 0,
        "errors": 0,
    }
    report = []

    for split in SPLITS:
        image_dir = SOURCE / "images" / split
        label_dir = SOURCE / "labels" / split

        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Missing image/label folder for {split}")

        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            key = (split, image_path.name)
            original_label = label_dir / f"{image_path.stem}.txt"

            if key in reviewed:
                row = reviewed[key]
                proposed_label = (
                    PROPOSED / split / f"{image_path.stem}.txt"
                )

                # Do not silently accept unresolved review decisions.
                status = row.get("ReviewStatus", "").strip().lower()
                decision = row.get("Decision", "").strip().lower()

                if status not in {"reviewed", "complete", "approved"}:
                    counts["skipped"] += 1
                    report.append(
                        [split, image_path.name, "SKIPPED",
                         "ReviewStatus is not completed"]
                    )
                    continue

                if decision not in {"use_proposed", "approved", "accept"}:
                    counts["skipped"] += 1
                    report.append(
                        [split, image_path.name, "SKIPPED",
                         "Decision does not approve proposed labels"]
                    )
                    continue

                label_to_use = proposed_label
                label_error = validate_label_file(label_to_use)

                if label_error:
                    counts["errors"] += 1
                    report.append(
                        [split, image_path.name, "ERROR", label_error]
                    )
                    continue

                counts["reviewed"] += 1
                action = "REVIEWED_LABEL"
            else:
                label_to_use = original_label
                label_error = validate_label_file(label_to_use)

                if label_error:
                    counts["skipped"] += 1
                    report.append(
                        [split, image_path.name, "SKIPPED", label_error]
                    )
                    continue

                action = "ORIGINAL_LABEL"

            destination_images = OUTPUT / "images" / split
            destination_labels = OUTPUT / "labels" / split
            destination_images.mkdir(parents=True, exist_ok=True)
            destination_labels.mkdir(parents=True, exist_ok=True)

            shutil.copy2(image_path, destination_images / image_path.name)
            shutil.copy2(
                label_to_use,
                destination_labels / f"{image_path.stem}.txt"
            )

            counts["copied"] += 1
            report.append([split, image_path.name, action, "OK"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "cleaning_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Split", "Image", "Action", "Details"])
        writer.writerows(report)

    print("\nDataset cleaning finished.")
    print(f"Output: {OUTPUT}")
    print(f"Images copied: {counts['copied']}")
    print(f"Reviewed labels used: {counts['reviewed']}")
    print(f"Files skipped: {counts['skipped']}")
    print(f"Errors: {counts['errors']}")
    print(f"Report: {OUTPUT / 'cleaning_report.csv'}")
    print("\nReview cleaning_report.csv before training.")


if __name__ == "__main__":
    main()
