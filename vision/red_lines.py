import cv2
import numpy as np

def angle_deg(x1, y1, x2, y2):
    return abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

def length(x1, y1, x2, y2):
    return np.hypot(x2 - x1, y2 - y1)

def find_red_lines(mask, frame_shape):
    h, w = frame_shape[:2]

    # ---- parametre ----
    MAX_ANGLE_HOR = 25
    MAX_ANGLE_VER = 25
    MIN_LENGTH = 100

    TOP_Y = 100
    BOTTOM_Y = h - 100

    LEFT_ZONE_RATIO = 0.40    # 40 % fra venstre
    RIGHT_ZONE_RATIO = 0.60   # sidste 40 %

    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=MIN_LENGTH,
        maxLineGap=30
    )

    if lines is None:
        return None

    best = {
        "top": None,
        "bottom": None,
        "left": None,
        "right": None
    }
    best_len = {k: 0 for k in best}

    for l in lines:
        x1, y1, x2, y2 = l[0]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        ang = angle_deg(x1, y1, x2, y2)
        ln = length(x1, y1, x2, y2)

        # ---------- TOP ----------
        if cy < TOP_Y and ang < MAX_ANGLE_HOR and ln > best_len["top"]:
            best["top"] = (x1, y1, x2, y2)
            best_len["top"] = ln

        # ---------- BOTTOM ----------
        if cy > BOTTOM_Y and ang < MAX_ANGLE_HOR and ln > best_len["bottom"]:
            best["bottom"] = (x1, y1, x2, y2)
            best_len["bottom"] = ln

        # ---------- LEFT ----------
        if (
            cx < LEFT_ZONE_RATIO * w
            and abs(ang - 90) < MAX_ANGLE_VER
            and ln > best_len["left"]
        ):
            best["left"] = (x1, y1, x2, y2)
            best_len["left"] = ln

        # ---------- RIGHT ----------
        if (
            cx > RIGHT_ZONE_RATIO * w
            and abs(ang - 90) < MAX_ANGLE_VER
            and ln > best_len["right"]
        ):
            best["right"] = (x1, y1, x2, y2)
            best_len["right"] = ln

    return best