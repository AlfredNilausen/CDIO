import cv2
import numpy as np

from red_mask import red_mask
from boundary_points import extract_boundary_points
from best_fit_lines import fit_frame_lines
from geometry import draw_model_line, intersect_horizontal_vertical

CAMERA_INDEX = 0
SMOOTH_ALPHA = 0.80


prev_corners = None


def smooth_point(prev_pt, new_pt, alpha=0.8):
    if prev_pt is None:
        return new_pt
    x = int(round(alpha * prev_pt[0] + (1 - alpha) * new_pt[0]))
    y = int(round(alpha * prev_pt[1] + (1 - alpha) * new_pt[1]))
    return (x, y)


def draw_corner_labels(frame, corners):
    for name, (x, y) in corners.items():
        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            frame,
            name,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


def make_board_mask(frame_shape, corners, margin_px=12):
    """
    Laver en fyldt polygonmaske for banen og trækker den lidt indad,
    så vi kun søger inde i rammen.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.array([
        corners["TL"],
        corners["TR"],
        corners["BR"],
        corners["BL"]
    ], dtype=np.int32)

    cv2.fillPoly(mask, [pts], 255)

    # Træk masken lidt indad så vi holder os inde i rammen
    if margin_px > 0:
        ksize = 2 * margin_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


def contour_circularity(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        return 0.0

    return 4 * np.pi * area / (perimeter * perimeter)


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("frame mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("white mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("orange mask", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # --------------------------------------------------
    # 1) Find banehjørner
    # --------------------------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_r = red_mask(hsv)

    boundary_points = extract_boundary_points(mask_r)
    models = fit_frame_lines(boundary_points)

    draw_model_line(display, models["top"], (255, 0, 0), 2)
    draw_model_line(display, models["bottom"], (0, 0, 255), 2)
    draw_model_line(display, models["left"], (0, 255, 0), 2)
    draw_model_line(display, models["right"], (0, 255, 255), 2)

    corners = None
    frame_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if all(models[k] is not None for k in ["top", "bottom", "left", "right"]):
        new_corners = {
            "TL": intersect_horizontal_vertical(models["top"], models["left"]),
            "TR": intersect_horizontal_vertical(models["top"], models["right"]),
            "BL": intersect_horizontal_vertical(models["bottom"], models["left"]),
            "BR": intersect_horizontal_vertical(models["bottom"], models["right"]),
        }

        if all(v is not None for v in new_corners.values()):
            if prev_corners is None:
                corners = new_corners
            else:
                corners = {
                    name: smooth_point(prev_corners[name], new_corners[name], SMOOTH_ALPHA)
                    for name in new_corners
                }

            prev_corners = corners

            draw_corner_labels(display, corners)
            frame_mask = make_board_mask(frame.shape, corners, margin_px=12)

    # --------------------------------------------------
    # 2) Bolddetektion – samme lyssettings som jeres kode
    # --------------------------------------------------
    hsv_ball = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White detection
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv_ball, lower_white, upper_white)

    # Orange detection
    lower_orange = np.array([10, 120, 120])
    upper_orange = np.array([30, 255, 255])
    mask_orange = cv2.inRange(hsv_ball, lower_orange, upper_orange)

    # Begræns søgningen til området inde i rammen
    mask_white = cv2.bitwise_and(mask_white, frame_mask)
    mask_orange = cv2.bitwise_and(mask_orange, frame_mask)

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)

    # Blur helps smooth reflections
    mask_white = cv2.GaussianBlur(mask_white, (9, 9), 0)
    mask_orange = cv2.GaussianBlur(mask_orange, (9, 9), 0)

    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --------------------------------------------------
    # 3) Hvide bolde
    # --------------------------------------------------
    for c in contours_white:
        area = cv2.contourArea(c)
        if area < 150:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = contour_circularity(c)

        if circularity > 0.7:
            (x, y), radius = cv2.minEnclosingCircle(c)

            center = (int(x), int(y))
            radius = int(radius)

            cv2.circle(display, center, radius, (0, 255, 0), 2)
            cv2.circle(display, center, 4, (255, 0, 0), -1)

            text = f"White {center}"
            cv2.putText(
                display,
                text,
                (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # --------------------------------------------------
    # 4) Orange bolde
    # --------------------------------------------------
    for c in contours_orange:
        area = cv2.contourArea(c)
        if area < 150:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = contour_circularity(c)

        if circularity > 0.7:
            (x, y), radius = cv2.minEnclosingCircle(c)

            center = (int(x), int(y))
            radius = int(radius)

            cv2.circle(display, center, radius, (0, 165, 255), 3)
            cv2.circle(display, center, 4, (255, 0, 0), -1)

            text = f"Orange {center}"
            cv2.putText(
                display,
                text,
                (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2
            )

    # --------------------------------------------------
    # 5) Visning
    # --------------------------------------------------
    cv2.imshow("frame mask", frame_mask)
    cv2.imshow("white mask", mask_white)
    cv2.imshow("orange mask", mask_orange)
    cv2.imshow("camera", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()