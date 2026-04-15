import cv2
import numpy as np


def intersect_horizontal_vertical(h_model, v_model):
    """
    h_model: y = mh*x + bh
    v_model: x = mv*y + bv
    """
    mh = h_model["m"]
    bh = h_model["b"]

    mv = v_model["m"]
    bv = v_model["b"]

    denom = 1.0 - mv * mh
    if abs(denom) < 1e-9:
        return None

    x = (mv * bh + bv) / denom
    y = mh * x + bh

    return int(round(x)), int(round(y))


def draw_model_line(frame, model, color, thickness=3):
    if model is None:
        return

    h, w = frame.shape[:2]

    if model["mode"] == "y_from_x":
        # y = m*x + b
        x1 = 0
        y1 = int(round(model["m"] * x1 + model["b"]))
        x2 = w - 1
        y2 = int(round(model["m"] * x2 + model["b"]))

    else:
        # x = m*y + b
        y1 = 0
        x1 = int(round(model["m"] * y1 + model["b"]))
        y2 = h - 1
        x2 = int(round(model["m"] * y2 + model["b"]))

    ok, p1, p2 = cv2.clipLine((0, 0, w, h), (x1, y1), (x2, y2))
    if ok:
        cv2.line(frame, p1, p2, color, thickness)