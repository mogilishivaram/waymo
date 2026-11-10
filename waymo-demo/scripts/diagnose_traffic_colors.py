
from pathlib import Path
from ultralytics import YOLO
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "runs/detect/use_cases/traffic_light_detection/results/s2tld_baseline/weights/best.pt"
DATA_YAML = ROOT / "datasets/traffic_lights/cleaned/data.yaml"
TEST_IMAGES = ROOT / "datasets/traffic_lights/cleaned/images/test"
TEST_LABELS = ROOT / "datasets/traffic_lights/cleaned/labels/test"

model = YOLO(str(MODEL_PATH))

with open(DATA_YAML, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("MODEL NAMES:", model.names)
print("DATASET NAMES:", config["names"])

# Show a few labeled images for a manual comparison.
image_files = sorted(
    p for p in TEST_IMAGES.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

shown = 0

for image_path in image_files:
    label_path = TEST_LABELS / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    ground_truth = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            class_id = int(float(parts[0]))
            ground_truth.append(class_id)

    if not ground_truth:
        continue

    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.15,
        device=0,
        verbose=False
    )[0]

    predicted = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        predicted.append((class_id, round(confidence, 3)))

    print("\nIMAGE:", image_path.name)
    print("GROUND-TRUTH CLASS IDS:", ground_truth)
    print("PREDICTIONS (class_id, confidence):", predicted)

    shown += 1
    if shown >= 15:
        break

print(f"\nInspected {shown} labeled test images.")