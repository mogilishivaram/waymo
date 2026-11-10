
from pathlib import Path
import csv
import json
import cv2

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "traffic_lights" / "coco-s2tld"
REVIEW_ROOT = PROJECT_ROOT / "datasets" / "traffic_lights" / "processed_review"
MANIFEST = REVIEW_ROOT / "annotation_review.csv"
OUTPUT_ROOT = REVIEW_ROOT / "proposed_labels"

# PROVISIONAL mapping. Verify before training.
CLASS_NAMES = {
    0: "red",
    1: "yellow",
    2: "green",
    3: "off",
    4: "wait_on",
}

WINDOW = "S2TLD Annotation Review"
CANVAS_W, CANVAS_H = 1400, 900


def load_manifest():
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}\n"
            "Create annotation_review.csv before launching this tool."
        )

    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("The annotation review manifest is empty.")

    return rows


def get_image_path(row):
    # Prefer the path recorded in the manifest.
    candidate = Path(row.get("ImagePath", ""))

    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    if candidate.exists():
        return candidate

    # Fallback for manifests containing only split and image name.
    candidate = (
        DATASET_ROOT / "images" / row["Split"] / row["Image"]
    )

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Image not found: {row.get('Image', '<unknown>')}"
    )


def get_output_paths(row):
    stem = Path(row["Image"]).stem
    folder = OUTPUT_ROOT / row["Split"]
    return folder / f"{stem}.txt", folder / f"{stem}.json"


def load_existing_boxes(row):
    _, json_path = get_output_paths(row)

    if not json_path.exists():
        return []

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data.get("boxes", [])
    except (OSError, json.JSONDecodeError):
        return []


def update_manifest(rows):
    fields = list(dict.fromkeys(
        key for row in rows for key in row.keys()
    ))

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_review(rows, index, boxes, status, notes):
    row = rows[index]
    image = cv2.imread(str(get_image_path(row)))

    if image is None:
        raise RuntimeError(f"Could not read {row['Image']}")

    height, width = image.shape[:2]
    txt_path, json_path = get_output_paths(row)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    yolo_lines = []
    valid_boxes = []

    for box in boxes:
        class_id = int(box["class_id"])
        x1, y1, x2, y2 = map(float, box["xyxy"])

        if class_id not in CLASS_NAMES:
            continue

        # Clip boxes to image bounds.
        x1 = max(0.0, min(x1, width - 1.0))
        x2 = max(0.0, min(x2, width - 1.0))
        y1 = max(0.0, min(y1, height - 1.0))
        y2 = max(0.0, min(y2, height - 1.0))

        if x2 <= x1 or y2 <= y1:
            continue

        # Convert pixel coordinates to normalized YOLO format.
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height

        yolo_lines.append(
            f"{class_id} {cx:.8f} {cy:.8f} "
            f"{bw:.8f} {bh:.8f}"
        )

        valid_boxes.append({
            "class_id": class_id,
            "xyxy": [x1, y1, x2, y2],
        })

    # Never write to the original labels directory.
    txt_path.write_text(
        "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
        encoding="utf-8",
    )

    json_path.write_text(
        json.dumps({
            "image": row["Image"],
            "split": row["Split"],
            "status": status,
            "notes": notes,
            "class_mapping_provisional": CLASS_NAMES,
            "boxes": valid_boxes,
            "warning": (
                "Proposed labels only. Verify class mapping and "
                "annotation policy before training."
            ),
        }, indent=2),
        encoding="utf-8",
    )

    row["ReviewStatus"] = status
    row["Decision"] = {
        "annotated": "Proposed annotations saved",
        "negative": "Reviewed as negative",
        "exclude": "Exclude from processed dataset",
        "pending": "Pending review",
    }[status]
    row["Notes"] = notes

    update_manifest(rows)

    print(f"Saved proposed labels: {txt_path}")
    print(f"Saved review metadata: {json_path}")


class AnnotationApp:
    def __init__(self, rows):
        self.rows = rows
        self.index = 0
        self.image = None
        self.boxes = []
        self.selected_class = 0
        self.scale = 1.0
        self.dragging = False
        self.start = None
        self.end = None
        self.message = ""
        self.notes = ""

    def load_image(self):
        row = self.rows[self.index]
        self.image = cv2.imread(str(get_image_path(row)))

        if self.image is None:
            raise RuntimeError(f"Could not open image: {row['Image']}")

        self.boxes = load_existing_boxes(row)
        self.notes = row.get("Notes", "")
        self.message = f"Loaded {row['Image']}"
        self.render()

    def to_original(self, point):
        height, width = self.image.shape[:2]

        x = int(point[0] / self.scale)
        y = int(point[1] / self.scale)

        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        return x, y

    def render(self):
        if self.image is None:
            return

        height, width = self.image.shape[:2]

        self.scale = min(
            (CANVAS_W - 20) / width,
            (CANVAS_H - 150) / height,
            1.0,
        )

        display_w = int(width * self.scale)
        display_h = int(height * self.scale)

        view = cv2.resize(
            self.image,
            (display_w, display_h),
            interpolation=cv2.INTER_AREA,
        )

        canvas = cv2.copyMakeBorder(
            view,
            0,
            CANVAS_H - 120 - display_h,
            0,
            CANVAS_W - display_w,
            cv2.BORDER_CONSTANT,
            value=(30, 30, 30),
        )

        # Draw existing proposed boxes.
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = map(float, box["xyxy"])

            p1 = (
                int(x1 * self.scale),
                int(y1 * self.scale),
            )
            p2 = (
                int(x2 * self.scale),
                int(y2 * self.scale),
            )

            color = (0, 255, 0) if i < len(self.boxes) - 1 else (
                0, 220, 255
            )

            cv2.rectangle(canvas, p1, p2, color, 2)

            class_id = int(box["class_id"])
            label = f"{class_id}: {CLASS_NAMES[class_id]}"

            cv2.putText(
                canvas,
                label,
                (p1[0], max(18, p1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Draw box currently being dragged.
        if self.dragging and self.start and self.end:
            cv2.rectangle(
                canvas, self.start, self.end, (255, 255, 0), 2
            )

        row = self.rows[self.index]

        messages = [
            f"{self.index + 1}/{len(self.rows)} | "
            f"{row['Split']} | {row['Image']}",
            f"Class {self.selected_class}: "
            f"{CLASS_NAMES[self.selected_class]} | "
            f"Boxes: {len(self.boxes)}",
            "Drag mouse=draw | 0-4=select class | S=save",
            "Backspace=undo | N=next | P=previous | "
            "E=negative | X=exclude | U=pending | Q=quit",
            "Mapping is provisional. Proposals are NOT training-ready.",
            self.message[:150],
        ]

        y = CANVAS_H - 115

        for message in messages:
            cv2.putText(
                canvas,
                message,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            y += 20

        cv2.imshow(WINDOW, canvas)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.end = (x, y)
            self.render()

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
            self.render()

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.end = (x, y)
            self.dragging = False

            x1, y1 = self.to_original(self.start)
            x2, y2 = self.to_original(self.end)

            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            if x2 - x1 >= 2 and y2 - y1 >= 2:
                self.boxes.append({
                    "class_id": self.selected_class,
                    "xyxy": [x1, y1, x2, y2],
                })
                self.message = "Box added. Press S to save."
            else:
                self.message = "Box too small; draw a larger rectangle."

            self.start = None
            self.end = None
            self.render()

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, CANVAS_W, CANVAS_H)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        self.load_image()

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                break

            elif ord("0") <= key <= ord("4"):
                self.selected_class = key - ord("0")
                self.message = (
                    f"Selected {self.selected_class}: "
                    f"{CLASS_NAMES[self.selected_class]}"
                )

            elif key in (8, 127):
                if self.boxes:
                    self.boxes.pop()
                    self.message = "Removed last box."

            elif key == ord("s"):
                status = "annotated" if self.boxes else "pending"
                save_review(
                    self.rows, self.index, self.boxes,
                    status, self.notes,
                )
                self.message = "Saved proposed labels separately."

            elif key == ord("e"):
                self.boxes = []
                save_review(
                    self.rows, self.index, [], "negative",
                    "Marked negative during manual review; "
                    "verify annotation rules.",
                )
                self.message = "Saved negative review decision."

            elif key == ord("x"):
                save_review(
                    self.rows, self.index, self.boxes, "exclude",
                    "Excluded from processed dataset pending review.",
                )
                self.message = "Marked for exclusion."

            elif key == ord("u"):
                save_review(
                    self.rows, self.index, self.boxes, "pending",
                    "Needs further review.",
                )
                self.message = "Marked pending."

            elif key == ord("n"):
                self.index = (self.index + 1) % len(self.rows)
                self.load_image()
                continue

            elif key == ord("p"):
                self.index = (self.index - 1) % len(self.rows)
                self.load_image()
                continue

            self.render()

        cv2.destroyAllWindows()


def main():
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()

    print(f"Loaded {len(rows)} review items.")
    print(f"Manifest: {MANIFEST}")
    print("Original S2TLD labels will not be modified.")
    print("Class mapping is provisional; do not train from proposals yet.")

    AnnotationApp(rows).run()


if __name__ == "__main__":
    main()
