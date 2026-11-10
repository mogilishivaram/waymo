from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]

model_path = (
    ROOT / "use_cases" / "object_tracking" / "models"
    / "waymo_vehicle_baseline-3" / "weights" / "best.pt"
)
source_path = ROOT / "data" / "images"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

model = YOLO(str(model_path))

model.predict(
    source=str(source_path),
    imgsz=640,
    conf=0.25,
    device=0,
    save=True,
    project=str(ROOT / "use_cases" / "object_tracking" / "results"),
    name="vehicle_detection",
)

print("Detection complete.")
print("Check:", ROOT / "use_cases" / "object_tracking" / "results" / "vehicle_detection")
