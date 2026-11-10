
from pathlib import Path

import cv2
import numpy as np


# Project paths
ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = ROOT / "data" / "images"
OUTPUT_DIR = ROOT / "outputs" / "lane_detection"

VIDEO_PATH = OUTPUT_DIR / "lane_detection_baseline.mp4"
PREVIEW_PATH = OUTPUT_DIR / "lane_detection_preview.jpg"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"

FPS = 10


def region_of_interest(image):
    """Keep the lower road region and ignore much of the sky/buildings."""
    height, width = image.shape[:2]

    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.35), int(height * 0.48)),
        (int(width * 0.65), int(height * 0.48)),
        (int(width * 0.95), height),
    ]], dtype=np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(image, mask)


def detect_lanes(frame):
    """Detect candidate lane markings using color, edges, and Hough lines."""
    height, width = frame.shape[:2]

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # White lane markings
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 155]),
        np.array([180, 90, 255])
    )

    # Yellow lane markings
    yellow_mask = cv2.inRange(
        hsv,
        np.array([12, 55, 80]),
        np.array([40, 255, 255])
    )

    # Detect road edges independently of color
    edges = cv2.Canny(gray, 50, 150)

    # Combine color candidates and edges.
    # OR is less restrictive than requiring a pixel to satisfy both.
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    combined = cv2.bitwise_or(color_mask, edges)

    roi = region_of_interest(combined)

    # Connect small gaps in candidate markings
    kernel = np.ones((3, 3), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

    segments = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=25,
        maxLineGap=40
    )

    # Draw individual candidate segments in orange
    segment_overlay = frame.copy()
    left_points = []
    right_points = []

    if segments is not None:
        for segment in segments:
            coords = np.asarray(segment).reshape(-1)

            if coords.size != 4:
                continue

            x1, y1, x2, y2 = map(int, coords)

            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 3:
                continue

            slope = dy / dx

            # Ignore nearly horizontal edges
            if abs(slope) < 0.35:
                continue

            # Exclude segments mostly outside the road's lower region
            if max(y1, y2) < int(height * 0.48):
                continue

            cv2.line(
                segment_overlay,
                (x1, y1),
                (x2, y2),
                (0, 165, 255),
                2
            )

            # Group left- and right-leaning line candidates
            midpoint_x = (x1 + x2) / 2

            if slope < 0 and midpoint_x < width * 0.60:
                left_points.extend([(x1, y1), (x2, y2)])

            elif slope > 0 and midpoint_x > width * 0.40:
                right_points.extend([(x1, y1), (x2, y2)])

    # Fit one representative line to each group
    lane_overlay = frame.copy()
    lane_found = False

    def fit_lane_line(points, is_left):
        if len(points) < 4:
            return None

        points = np.array(points, dtype=np.float32)
        xs = points[:, 0]
        ys = points[:, 1]

        # x = m*y + b works well for near-vertical lane boundaries
        try:
            m, b = np.polyfit(ys, xs, 1)
        except (ValueError, np.linalg.LinAlgError):
            return None

        y_bottom = height - 1
        y_top = int(height * 0.55)

        x_bottom = int(m * y_bottom + b)
        x_top = int(m * y_top + b)

        # Reject implausible fitted lines outside the image
        if not (-width * 0.25 <= x_bottom <= width * 1.25):
            return None

        if not (-width * 0.25 <= x_top <= width * 1.25):
            return None

        # Check the expected side of the image
        if is_left and x_bottom > width * 0.65:
            return None

        if not is_left and x_bottom < width * 0.35:
            return None

        return (x_bottom, y_bottom), (x_top, y_top)

    left_line = fit_lane_line(left_points, True)
    right_line = fit_lane_line(right_points, False)

    if left_line:
        cv2.line(
            lane_overlay, left_line[0], left_line[1],
            (0, 255, 0), 5
        )
        lane_found = True

    if right_line:
        cv2.line(
            lane_overlay, right_line[0], right_line[1],
            (0, 255, 0), 5
        )
        lane_found = True

    # Clearly label the result rather than implying every frame succeeded
    if lane_found:
        status = "CANDIDATE LANE LINE(S) DETECTED"
        status_color = (0, 255, 0)
    else:
        status = "NO RELIABLE LANE CANDIDATES"
        status_color = (0, 0, 255)

    cv2.putText(
        lane_overlay,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        2,
        cv2.LINE_AA
    )

    return {
        "overlay": lane_overlay,
        "segments": segment_overlay,
        "white_mask": white_mask,
        "yellow_mask": yellow_mask,
        "edges": edges,
        "combined": combined,
        "roi": roi,
        "lane_found": lane_found,
    }


def save_diagnostics(result):
    """Save intermediate images from the first frame."""
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    images = {
        "01_white_mask.jpg": result["white_mask"],
        "02_yellow_mask.jpg": result["yellow_mask"],
        "03_edges.jpg": result["edges"],
        "04_combined_candidates.jpg": result["combined"],
        "05_road_region.jpg": result["roi"],
        "06_hough_segments.jpg": result["segments"],
        "07_lane_overlay.jpg": result["overlay"],
    }

    for filename, image in images.items():
        path = DIAGNOSTICS_DIR / filename
        if not cv2.imwrite(str(path), image):
            print(f"Warning: Could not save {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(INPUT_DIR.glob("front_*.jpg"))

    if not image_paths:
        print(f"No front_*.jpg images found in: {INPUT_DIR}")
        print("Check the INPUT_DIR path and your extracted images.")
        return

    first_frame = cv2.imread(str(image_paths[0]))

    if first_frame is None:
        print(f"Could not read image: {image_paths[0]}")
        return

    height, width = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        str(VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height)
    )

    if not writer.isOpened():
        print("Could not create output video.")
        print("Check that OpenCV supports the MP4V codec.")
        return

    detected_count = 0
    processed_count = 0

    try:
        for index, image_path in enumerate(image_paths, start=1):
            frame = cv2.imread(str(image_path))

            if frame is None:
                print(f"Skipping unreadable image: {image_path.name}")
                continue

            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))

            result = detect_lanes(frame)
            writer.write(result["overlay"])

            if result["lane_found"]:
                detected_count += 1

            if processed_count == 0:
                cv2.imwrite(str(PREVIEW_PATH), result["overlay"])
                save_diagnostics(result)

            processed_count += 1

            if index % 25 == 0 or index == len(image_paths):
                print(
                    f"Processed {index}/{len(image_paths)} | "
                    f"frames with candidate lanes: {detected_count}"
                )

    finally:
        writer.release()

    print("\nLane detection baseline finished.")
    print(f"Images processed: {processed_count}")
    print(f"Frames with candidate lane lines: {detected_count}")
    print(f"Preview: {PREVIEW_PATH}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Diagnostics: {DIAGNOSTICS_DIR}")


if __name__ == "__main__":
    main()
