from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "use_cases" / "object_tracking" / "models"
DATASET_CONFIG = ROOT / "datasets" / "object_tracking" / "data.yaml"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Training stopped.")

    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(str(MODELS_DIR / "yolo11n.pt"))

    model.train(
        data=str(DATASET_CONFIG),
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project=str(MODELS_DIR),
        name="waymo_vehicle_baseline",
        patience=10,
        save=True,
        plots=True,
    )

    print("Training finished.")

if __name__ == "__main__":
    freeze_support()
    main()
