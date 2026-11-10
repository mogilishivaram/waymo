# Object tracking

- **Algorithm:** YOLOv8 object detection with ByteTrack tracking.
- **Dataset:** The preserved Waymo extraction is in `data/images`; the prepared
  YOLO dataset remains in `data/yolo_vehicle` and is referenced by
  `datasets/object_tracking/data.yaml` without copying it.
- **Preserved artifacts:** `models/yolov8n.pt`, `models/yolo11n.pt`, and the
  trained YOLO11 experiment under `models/waymo_vehicle_baseline-3`.
- **Status:** Inference and the existing tracking demo are available. Training
  remains an explicit, GPU-dependent operation.

From the project root, run
`python use_cases/object_tracking/scripts/track_vehicles.py` to regenerate the
object-tracking video. Run
`python use_cases/object_tracking/scripts/predict_vehicle.py` for saved
detection images. The YOLO11 training script preserves the original experiment
and is not the final YOLOv8 + ByteTrack pipeline.
