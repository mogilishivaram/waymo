
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "datasets/traffic_lights/coco-s2tld/images/val/2020-03-30 12_02_12.042615407.jpg"
OUTPUT = ROOT / "datasets/traffic_lights/processed_review/proposed_labels/val/2020-03-30 12_02_12.042615407.txt"

CLASSES = {
    0: "red",
    1: "yellow",
    2: "green",
    3: "off",
    4: "wait_on",
}

COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 255),
    2: (0, 255, 0),
    3: (200, 200, 200),
    4: (255, 255, 0),
}

image = cv2.imread(str(IMAGE))
if image is None:
    raise FileNotFoundError(f"Cannot open image: {IMAGE}")

height, width = image.shape[:2]
boxes = []
selected_class = 0
start = None
preview = None


def mouse(event, x, y, flags, param):
    global start, preview

    if event == cv2.EVENT_LBUTTONDOWN:
        start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and start is not None:
        preview = (start[0], start[1], x, y)

    elif event == cv2.EVENT_LBUTTONUP and start is not None:
        x1, y1 = start
        x2, y2 = x, y

        x1, x2 = sorted((max(0, min(width - 1, x1)),
                         max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)),
                         max(0, min(height - 1, y2))))

        if x2 > x1 and y2 > y1:
            boxes.append((selected_class, x1, y1, x2, y2))
            print(f"Added {CLASSES[selected_class]} box: "
                  f"{x1},{y1} to {x2},{y2}")

        start = None
        preview = None


def render():
    canvas = image.copy()

    for class_id, x1, y1, x2, y2 in boxes:
        color = COLORS[class_id]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas, CLASSES[class_id], (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    if preview and start:
        cv2.rectangle(
            canvas, start, (preview[2], preview[3]),
            COLORS[selected_class], 1
        )

    help_text = (
        f"Class: {selected_class}={CLASSES[selected_class]} | "
        "0-4 select class | S save | U undo | Q quit"
    )
    cv2.rectangle(canvas, (0, 0), (width, 32), (30, 30, 30), -1)
    cv2.putText(
        canvas, help_text, (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
    )
    return canvas


print("Annotating:", IMAGE)
print("Draw boxes tightly around visible traffic-light indicators.")
print("Keys: 0-4 select class | U undo | S save | Q quit")

cv2.namedWindow("Annotate Fourth Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Annotate Fourth Image", 1200, 750)
cv2.setMouseCallback("Annotate Fourth Image", mouse)

while True:
    cv2.imshow("Annotate Fourth Image", render())
    key = cv2.waitKey(20) & 0xFF

    if ord("0") <= key <= ord("4"):
        selected_class = key - ord("0")
        print("Selected:", CLASSES[selected_class])

    elif key in (ord("u"), ord("U")):
        if boxes:
            boxes.pop()
            print("Last box removed.")

    elif key in (ord("s"), ord("S")):
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for class_id, x1, y1, x2, y2 in boxes:
            xc = ((x1 + x2) / 2) / width
            yc = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            lines.append(
                f"{class_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}"
            )

        OUTPUT.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8"
        )
        print(f"Saved {len(boxes)} boxes to:\n{OUTPUT}")

    elif key in (ord("q"), ord("Q"), 27):
        break

cv2.destroyAllWindows()
