from pathlib import Path
import cv2


def image_paths(directory: Path, pattern: str = "*.jpg") -> list[Path]:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No images matching {pattern!r} found in {directory}")
    return paths


def open_video_writer(path: Path, frame_size: tuple[int, int], fps: float = 10.0) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    return writer
