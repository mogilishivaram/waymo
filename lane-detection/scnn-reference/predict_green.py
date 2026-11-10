
import argparse
import cv2
import numpy as np
import torch

from model import SCNN
from utils.transforms import Resize, Compose, ToTensor, Normalize


def parse_args():
    parser = argparse.ArgumentParser(
        description="SCNN lane detection with green lane overlay"
    )
    parser.add_argument("--img_path", required=True)
    parser.add_argument("--weight_path", required=True)
    parser.add_argument(
        "--output",
        default="demo/demo_result_green.jpg",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Read original image.
    original = cv2.imread(args.img_path)

    if original is None:
        raise FileNotFoundError(
            f"Cannot read image: {args.img_path}"
        )

    original_h, original_w = original.shape[:2]
    print(f"Original image: {original_w} x {original_h}")

    # Preprocess using the SCNN repository's expected settings.
    image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    resized = Resize((800, 288))(
        {"img": image_rgb}
    )["img"]

    transform = Compose(
        ToTensor(),
        Normalize(
            mean=(0.3598, 0.3653, 0.3662),
            std=(0.2573, 0.2663, 0.2756),
        ),
    )

    tensor = transform({"img": resized})["img"]
    tensor = tensor.unsqueeze(0).to(device)

    # Load SCNN and checkpoint.
    model = SCNN(
        input_size=(800, 288),
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(
        args.weight_path,
        map_location=device,
        weights_only=False,
    )

    if "net" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain the expected 'net' key."
        )

    model.load_state_dict(checkpoint["net"])
    model.eval()

    # Inference.
    with torch.no_grad():
        predictions = model(tensor)
        seg_pred, exist_pred = predictions[:2]

    labels = seg_pred[0].argmax(dim=0).cpu().numpy()
    exists = (
        exist_pred[0] > 0.5
    ).cpu().numpy()

    mask_h, mask_w = labels.shape

    print(f"Prediction mask: {mask_w} x {mask_h}")
    print("Lane existence scores:",
          exist_pred[0].cpu().numpy().round(3).tolist())
    print("Detected lane classes:", np.unique(labels).tolist())

    # Convert every detected lane class into points.
    candidates = []

    for lane_id in range(1, 5):
        if not exists[lane_id - 1]:
            print(f"Lane class {lane_id}: absent")
            continue

        ys, xs = np.where(labels == lane_id)

        if len(xs) < 10:
            print(f"Lane class {lane_id}: too few pixels")
            continue

        # Estimate lane x-position at each occupied image row.
        points = []

        for y in np.unique(ys):
            row_x = xs[ys == y]
            points.append(
                (float(np.median(row_x)), int(y))
            )

        # Use lower-image points to estimate the lane's position.
        bottom_points = [
            (x, y)
            for x, y in points
            if y >= mask_h * 0.65
        ]

        if not bottom_points:
            print(f"Lane class {lane_id}: no lower-image points")
            continue

        x_bottom = float(
            np.median([x for x, _ in bottom_points])
        )

        candidates.append({
            "lane_id": lane_id,
            "x_bottom": x_bottom,
            "points": points,
        })

        print(
            f"Lane class {lane_id}: "
            f"x_bottom={x_bottom:.1f}, "
            f"pixels={len(xs)}"
        )

    if len(candidates) < 2:
        raise RuntimeError(
            "SCNN returned fewer than two usable lane boundaries. "
            "Review the diagnostic output above."
        )

    # Choose the nearest available lane boundary on each side
    # of the segmentation map's center.
    center_x = mask_w / 2

    left_candidates = [
        lane for lane in candidates
        if lane["x_bottom"] < center_x
    ]

    right_candidates = [
        lane for lane in candidates
        if lane["x_bottom"] >= center_x
    ]

    # If the center-based split fails, use the two innermost
    # detected lane candidates as a fallback.
    if left_candidates and right_candidates:
        left_lane = max(
            left_candidates,
            key=lambda lane: lane["x_bottom"],
        )
        right_lane = min(
            right_candidates,
            key=lambda lane: lane["x_bottom"],
        )
    else:
        candidates.sort(
            key=lambda lane: lane["x_bottom"]
        )
        left_lane = candidates[0]
        right_lane = candidates[1]

        print(
            "Warning: center-based lane selection failed; "
            "using the two leftmost detected candidates."
        )

    print(
        f"Selected lane classes: "
        f"{left_lane['lane_id']} and "
        f"{right_lane['lane_id']}"
    )

    # Build row-to-x mappings for both selected boundaries.
    def make_row_map(lane):
        row_map = {}

        for x, y in lane["points"]:
            row_map[y] = x

        return row_map

    left_by_y = make_row_map(left_lane)
    right_by_y = make_row_map(right_lane)

    common_y = sorted(
        set(left_by_y) & set(right_by_y)
    )

    # Build polygon only where both boundaries exist and
    # the left boundary is actually left of the right boundary.
    polygon_left = []
    polygon_right = []

    for y in common_y:
        lx = left_by_y[y]
        rx = right_by_y[y]

        if lx < rx:
            polygon_left.append((lx, y))
            polygon_right.append((rx, y))

    if len(polygon_left) < 2:
        raise RuntimeError(
            "Selected lane boundaries do not form a valid "
            "overlapping polygon. Try a different road image."
        )

    # Convert segmentation coordinates to original-image coordinates.
    sx = original_w / mask_w
    sy = original_h / mask_h

    def scale_points(points):
        return np.array(
            [
                [
                    int(round(x * sx)),
                    int(round(y * sy)),
                ]
                for x, y in points
            ],
            dtype=np.int32,
        )

    polygon_points = scale_points(
        polygon_left + polygon_right[::-1]
    )

    # Shade the region between the lane boundaries.
    overlay = original.copy()

    cv2.fillPoly(
        overlay,
        [polygon_points],
        color=(0, 255, 0),
    )

    result = cv2.addWeighted(
        overlay,
        0.25,
        original,
        0.75,
        0,
    )

    # Draw both lane boundaries in bright green.
    for lane in (left_lane, right_lane):
        coords = scale_points(lane["points"])

        if len(coords) >= 2:
            cv2.polylines(
                result,
                [coords],
                isClosed=False,
                color=(0, 255, 0),
                thickness=4,
                lineType=cv2.LINE_AA,
            )

    # Save output.
    output_dir = (
        __import__("os").path.dirname(args.output)
    )

    if output_dir:
        __import__("os").makedirs(
            output_dir,
            exist_ok=True,
        )

    if not cv2.imwrite(args.output, result):
        raise RuntimeError(
            f"Could not save output: {args.output}"
        )

    print(f"Saved result: {args.output}")


if __name__ == "__main__":
    main()
