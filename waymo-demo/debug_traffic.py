
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(
    "runs/detect/runs/detect/traffic_light_detection_v2/weights/best.pt"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select the original traffic-light image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png"),
        ("All files", "*.*"),
    ],
)

root.destroy()

if not image_path:
    print("No image selected. Exiting.")
    raise SystemExit

print(f"\nSelected image: {image_path}")

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=image_path,
    conf=0.10,
    imgsz=960,
    device=0,
    save=True,
    project="runs/traffic_debug",
    name="same_image",
    exist_ok=True,
    verbose=True,
)

print("\nClass mapping:", model.names)
print("Detection count:", len(results[0].boxes))

for box in results[0].boxes:
    class_id = int(box.cls.item())
    class_name = model.names[class_id]
    confidence = float(box.conf.item())
    coordinates = box.xyxy[0].cpu().tolist()

    print(
        f"Class: {class_name} | "
        f"Confidence: {confidence:.3f} | "
        f"Box: {coordinates}"
    )
