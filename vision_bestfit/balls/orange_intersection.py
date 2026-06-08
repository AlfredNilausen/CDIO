# orange_intersection.py
import cv2

from orange_mask import compute_orange_mask
from white_closed import compute_white_closed


def compute_orange_intersection(frame):
    orange_mask = compute_orange_mask(frame)
    white_closed, _, _ = compute_white_closed(frame)

    # AND: orange farve + lukket form
    orange_intersection = cv2.bitwise_and(orange_mask, white_closed)

    return orange_intersection, orange_mask, white_closed
