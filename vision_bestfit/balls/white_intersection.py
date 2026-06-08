# white_intersection.py
import cv2
import numpy as np

from white_mask import compute_white_mask
from white_closed import compute_white_closed


def compute_white_intersection(frame):
    """
    Beholder kun pixels som er hvide i både:
    - white_mask (farve)
    - white_closed (lukket geometri)
    """

    white_mask = compute_white_mask(frame)
    white_closed, _, _ = compute_white_closed(frame)

    # INTERSECTION (AND)
    white_intersection = cv2.bitwise_and(white_mask, white_closed)

    return white_intersection, white_mask, white_closed


# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("white_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("white_closed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("white_intersection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        white_intersection, white_mask, white_closed = compute_white_intersection(frame)

        cv2.imshow("camera", frame)
        cv2.imshow("white_mask", white_mask)
        cv2.imshow("white_closed", white_closed)
        cv2.imshow("white_intersection", white_intersection)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
