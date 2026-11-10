
from pathlib import Path
from collections import Counter
import csv

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "runs" / "detect" / "use_cases"
    / "traffic_light_detection" / "results"
    / "s2tld_baseline" / "weights" / "best.pt"
)

IMAGE_DIR = ROOT / "datasets" / "traffic_lights" / "cleaned" / "images" / "test"
LABEL_DIR = ROOT / "datasets" / "traffic_lights" / "cleaned" / "labels" / "test"

OUTPUT_DIR = ROOT / "runs" / "traffic_color_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE = 0.15
IOU_THRESHOLD = 0.50

NAMES = {
    0: "red",
    1: "yellow",
    2: "green",
    3: "off",
    4: "wait_on",
}


def box_iou(a, b):
    """IoU between two xyxy bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def load_ground_truth(label_path, width, height):
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])

        # YOLO normalized xywh -> pixel xyxy
        boxes.append({
            "class": cls,
            "box": (
                (x - w / 2) * width,
                (y - h / 2) * height,
                (x + w / 2) * width,
                (y + h / 2) * height,
            ),
        })

    return boxes


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        raise FileNotFoundError("Test image or label folder not found.")

    model = YOLO(str(MODEL_PATH))
    image_files = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not image_files:
        raise RuntimeError(f"No test images found in {IMAGE_DIR}")

    totals = Counter()
    class_counts = Counter()
    confusion = Counter()
    report_rows = []

    for index, image_path in enumerate(image_files, start=1):
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        result = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=CONFIDENCE,
            device=0,
            verbose=False,
        )[0]

        width, height = result.orig_shape[1], result.orig_shape[0]

        gt = load_ground_truth(label_path, width, height)

        predictions = []
        for box in result.boxes:
            predictions.append({
                "class": int(box.cls.item()),
                "confidence": float(box.conf.item()),
                "box": tuple(float(v) for v in box.xyxy[0].tolist()),
            })

        # Candidate matches are class-agnostic. This is intentional:
        # we want to detect color mistakes on the correct physical object.
        candidates = []
        for gi, g in enumerate(gt):
            for pi, p in enumerate(predictions):
                iou = box_iou(g["box"], p["box"])
                if iou >= IOU_THRESHOLD:
                    candidates.append((iou, gi, pi))

        # Greedy one-to-one matching, highest IoU first.
        candidates.sort(reverse=True)
        matched_gt = set()
        matched_pred = set()

        for iou, gi, pi in candidates:
            if gi in matched_gt or pi in matched_pred:
                continue

            matched_gt.add(gi)
            matched_pred.add(pi)

            true_cls = gt[gi]["class"]
            pred_cls = predictions[pi]["class"]

            if true_cls == pred_cls:
                outcome = "correct_color"
                totals["correct_color"] += 1
                class_counts[(true_cls, "correct")] += 1
            else:
                outcome = "wrong_color"
                totals["wrong_color"] += 1
                class_counts[(true_cls, "wrong")] += 1

            confusion[(true_cls, pred_cls)] += 1

            report_rows.append({
                "image": image_path.name,
                "outcome": outcome,
                "true_class": NAMES.get(true_cls, str(true_cls)),
                "predicted_class": NAMES.get(pred_cls, str(pred_cls)),
                "confidence": round(predictions[pi]["confidence"], 4),
                "iou": round(iou, 4),
            })

        # Unmatched ground truths are missed detections.
        for gi, g in enumerate(gt):
            if gi not in matched_gt:
                totals["missed"] += 1
                class_counts[(g["class"], "missed")] += 1
                report_rows.append({
                    "image": image_path.name,
                    "outcome": "missed",
                    "true_class": NAMES.get(g["class"], str(g["class"])),
                    "predicted_class": "",
                    "confidence": "",
                    "iou": "",
                })

        # Unmatched predictions are false positives or duplicate boxes.
        for pi, p in enumerate(predictions):
            if pi not in matched_pred:
                totals["false_positive"] += 1
                report_rows.append({
                    "image": image_path.name,
                    "outcome": "false_positive",
                    "true_class": "",
                    "predicted_class": NAMES.get(p["class"], str(p["class"])),
                    "confidence": round(p["confidence"], 4),
                    "iou": "",
                })

        if index % 50 == 0 or index == len(image_files):
            print(f"Processed {index}/{len(image_files)} images")

    report_path = OUTPUT_DIR / "traffic_color_errors.csv"

    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image", "outcome", "true_class",
                "predicted_class", "confidence", "iou",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    matched = totals["correct_color"] + totals["wrong_color"]
    color_accuracy = (
        100 * totals["correct_color"] / matched if matched else 0
    )

    print("\n===== TRAFFIC-LIGHT EVALUATION =====")
    print(f"Images evaluated: {len(image_files)}")
    print(f"Confidence threshold: {CONFIDENCE}")
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print(f"Correct-color matches: {totals['correct_color']}")
    print(f"Wrong-color matches: {totals['wrong_color']}")
    print(f"Missed ground-truth lights: {totals['missed']}")
    print(f"Unmatched predictions: {totals['false_positive']}")
    print(f"Color accuracy among matched boxes: {color_accuracy:.2f}%")

    print("\n===== COLOR CONFUSIONS =====")
    if not confusion:
        print("No ground-truth boxes matched predictions.")
    else:
        for (true_cls, pred_cls), count in sorted(confusion.items()):
            print(
                f"Actual {NAMES.get(true_cls, true_cls)} -> "
                f"Predicted {NAMES.get(pred_cls, pred_cls)}: {count}"
            )

    print("\n===== PER-CLASS RESULTS =====")
    for cls, name in NAMES.items():
        correct = class_counts[(cls, "correct")]
        wrong = class_counts[(cls, "wrong")]
        missed = class_counts[(cls, "missed")]
        total = correct + wrong + missed

        if total:
            print(
                f"{name}: ground truth={total}, correct color={correct}, "
                f"wrong color={wrong}, missed={missed}"
            )

    print(f"\nDetailed report saved to:\n{report_path}")


if __name__ == "__main__":
    main()
