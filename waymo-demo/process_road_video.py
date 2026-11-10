from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

ROOT = Path(__file__).resolve().parent
TRAFFIC_MODEL_PATH = ROOT / "runs" / "detect" / "runs" / "detect" / "traffic_light_detection_v2" / "weights" / "best.pt"
OBJECT_MODEL_PATH = ROOT / "use_cases" / "object_tracking" / "models" / "waymo_vehicle_baseline-3" / "weights" / "best.pt"
DEVICE = 0
TRAFFIC_CONF = 0.25
OBJECT_CONF = 0.30
INFERENCE_SIZE = 640


def lane_overlay(frame):
    """Classical OpenCV lane-candidate overlay, not a trained lane model."""
    h, w = frame.shape[:2]
    output = frame.copy()
    y_top = int(h * 0.52)
    polygon = np.array([[
        (int(w * 0.04), h - 1), (int(w * 0.38), y_top),
        (int(w * 0.62), y_top), (int(w * 0.96), h - 1)
    ]], dtype=np.int32)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, polygon, 255)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    white = cv2.inRange(hls, np.array([0, 150, 0]), np.array([180, 255, 110]))
    yellow = cv2.inRange(hsv, np.array([12, 45, 80]), np.array([40, 255, 255]))
    color_candidates = cv2.bitwise_and(cv2.bitwise_or(white, yellow), roi_mask)

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    guided_edges = cv2.bitwise_and(
        edges, cv2.dilate(color_candidates, np.ones((3, 3), np.uint8), iterations=1)
    )
    candidates = cv2.morphologyEx(
        cv2.bitwise_or(color_candidates, guided_edges),
        cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )

    lines = cv2.HoughLinesP(
        candidates, 1, np.pi / 180, threshold=25,
        minLineLength=max(20, int(w * 0.025)), maxLineGap=max(20, int(w * 0.025))
    )
    lx, ly, rx, ry = [], [], [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = np.asarray(line).reshape(-1).astype(int)
            dx = x2 - x1
            if abs(dx) < 3:
                continue
            slope = (y2 - y1) / dx
            if abs(slope) < 0.35 or max(y1, y2) < y_top:
                continue
            midpoint = (x1 + x2) / 2
            if slope < 0 and midpoint < w * 0.60:
                lx.extend([x1, x2]); ly.extend([y1, y2])
            elif slope > 0 and midpoint > w * 0.40:
                rx.extend([x1, x2]); ry.extend([y1, y2])

    def fit(xs, ys, side):
        if len(xs) < 6:
            return None
        try:
            m, b = np.polyfit(np.asarray(ys, dtype=np.float32),
                              np.asarray(xs, dtype=np.float32), 1)
        except (ValueError, np.linalg.LinAlgError):
            return None
        y1, y2 = h - 1, y_top
        x1, x2 = int(m * y1 + b), int(m * y2 + b)
        if side == "left" and not (w * 0.05 <= x1 <= w * 0.65):
            return None
        if side == "right" and not (w * 0.35 <= x1 <= w * 0.95):
            return None
        if not (-w * 0.1 <= x2 <= w * 1.1):
            return None
        return (x1, y1), (x2, y2)

    left, right = fit(lx, ly, "left"), fit(rx, ry, "right")
    found = False
    thickness = max(3, w // 320)
    for lane in (left, right):
        if lane:
            cv2.line(output, lane[0], lane[1], (0, 255, 0), thickness)
            found = True
    label = "LANE CANDIDATES" if found else "LANES NOT CLEARLY DETECTED"
    color = (0, 255, 0) if found else (0, 165, 255)
    cv2.putText(output, label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return output


def choose_video():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select a driving video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    root.destroy()
    return Path(path) if path else None


def main():
    for label, path in (("Traffic-light", TRAFFIC_MODEL_PATH), ("Object", OBJECT_MODEL_PATH)):
        if not path.exists():
            raise FileNotFoundError(
                f"{label} model not found:\n{path}\n"
                "Update the corresponding model path near the top of this script."
            )

    video_path = choose_video()
    if video_path is None:
        print("No video selected. Exiting.")
        return
    output_path = video_path.with_name(video_path.stem + "_road_predictions.mp4")

    print("Loading traffic-light model...")
    traffic_model = YOLO(str(TRAFFIC_MODEL_PATH))
    print("Loading object model...")
    object_model = YOLO(str(OBJECT_MODEL_PATH))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output MP4. Check the MP4V codec.")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            annotated = lane_overlay(frame)

            object_result = object_model.predict(
                source=frame, conf=OBJECT_CONF, imgsz=INFERENCE_SIZE,
                device=DEVICE, verbose=False
            )[0]
            annotated = object_result.plot(img=annotated)

            traffic_result = traffic_model.predict(
                source=frame, conf=TRAFFIC_CONF, imgsz=INFERENCE_SIZE,
                device=DEVICE, verbose=False
            )[0]
            annotated = traffic_result.plot(img=annotated)

            writer.write(annotated)
            if frame_index == 1 or frame_index % 25 == 0:
                pct = 100 * frame_index / total if total > 0 else 0
                print(f"Processed {frame_index}/{total} frames ({pct:.1f}%)")
    finally:
        cap.release()
        writer.release()

    print("\nFinished.")
    print(f"Input video:  {video_path}")
    print(f"Output video: {output_path}")
    print("Lane markings are OpenCV candidates; validate them visually.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        input("Press Enter to close...")
