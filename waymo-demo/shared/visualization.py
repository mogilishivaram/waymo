import cv2


def draw_label(
    frame,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 0),
):
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame
