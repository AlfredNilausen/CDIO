import cv2
import numpy as np

from red_mask import red_mask
from boundary_points import extract_boundary_points
from best_fit_lines import fit_frame_lines
from geometry import draw_model_line, intersect_horizontal_vertical

CAMERA_INDEX = 0

# smoothing af hjørner
prev_corners = None
SMOOTH_ALPHA = 0.80  # højere = mere stabil, men lidt mere træg

def smooth_point(prev_pt, new_pt, alpha=0.8):
    if prev_pt is None:
        return new_pt
    x = int(round(alpha * prev_pt[0] + (1 - alpha) * new_pt[0]))
    y = int(round(alpha * prev_pt[1] + (1 - alpha) * new_pt[1]))
    return (x, y)

def draw_corner_info(frame, corners):
    """
    Tegner koordinater både ved hjørnerne og som tekstpanel øverst i billedet.
    """
    # tekstpanel
    y0 = 25
    dy = 25

    order = ["TL", "TR", "BL", "BR"]
    for i, name in enumerate(order):
        x, y = corners[name]
        text = f"{name}: ({x}, {y})"
        cv2.putText(
            frame,
            text,
            (20, y0 + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # labels ved hjørnerne
    for name, (x, y) in corners.items():
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"{name} ({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)

    boundary_points = extract_boundary_points(mask)
    models = fit_frame_lines(boundary_points)

    # -------- tegn rå randpunkter (debug) --------
    colors_points = {
        "top": (255, 0, 0),
        "bottom": (0, 0, 255),
        "left": (0, 255, 0),
        "right": (0, 255, 255),
    }

    for name, pts in boundary_points.items():
        if pts is not None and len(pts) > 0:
            for p in pts[::20]:   # tegn hver 20. pixel for debug
                x, y = int(p[0]), int(p[1])
                cv2.circle(display, (x, y), 1, colors_points[name], -1)

    # -------- tegn fit-linjer --------
    draw_model_line(display, models["top"], (255, 0, 0), 3)       # blå
    draw_model_line(display, models["bottom"], (0, 0, 255), 3)    # rød
    draw_model_line(display, models["left"], (0, 255, 0), 3)      # grøn
    draw_model_line(display, models["right"], (0, 255, 255), 3)   # gul

    corners = None

    # -------- beregn hjørner --------
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

            draw_corner_info(display, corners)

    cv2.imshow("camera", display)
    cv2.imshow("red mask", mask)

    key = cv2.waitKey(1) & 0xFF

    # ESC = afslut
    if key == 27:
        break

    # p = print hjørnekoordinater til terminal
    if key == ord("p") and corners is not None:
        print("----- Pixel coordinates in camera frame -----")
        print(f"TL = {corners['TL']}")
        print(f"TR = {corners['TR']}")
        print(f"BL = {corners['BL']}")
        print(f"BR = {corners['BR']}")
        print("--------------------------------------------")

cap.release()
cv2.destroyAllWindows()