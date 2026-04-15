import cv2
import numpy as np

from red_mask import red_mask
from boundary_points import extract_boundary_points
from best_fit_lines import fit_frame_lines
from geometry import draw_model_line, intersect_horizontal_vertical
from ball_masks import (
    make_board_mask,
    make_inner_play_area_mask,
    get_orange_ball_mask,
    get_white_ball_mask,
    find_ball_candidates
)

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
            f"{name}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("red mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("orange mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("white mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("board mask", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # ---------------- FIND BOARD ----------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_r = red_mask(hsv)

    boundary_points = extract_boundary_points(mask_r)
    models = fit_frame_lines(boundary_points)

    draw_model_line(display, models["top"], (255, 0, 0), 2)
    draw_model_line(display, models["bottom"], (0, 0, 255), 2)
    draw_model_line(display, models["left"], (0, 255, 0), 2)
    draw_model_line(display, models["right"], (0, 255, 255), 2)

    corners = None

    corners = None
    board_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    play_area_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

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
            board_mask = make_board_mask(frame.shape, corners)
            play_area_mask = make_inner_play_area_mask(frame.shape, corners, mask_r, margin_px=25)

    # ---------------- BALL MASKS ----------------
    orange_mask = get_orange_ball_mask(frame, roi_mask=play_area_mask)
    white_mask = get_white_ball_mask(frame, roi_mask=play_area_mask)

    orange_balls = find_ball_candidates(
        orange_mask,
        min_area=20,
        max_area=1200,
        min_circularity=0.50
    )

    white_balls = find_ball_candidates(
        white_mask,
        min_area=20,
        max_area=1200,
        min_circularity=0.50
    )

    # ---------------- DRAW RESULTS ----------------
    for ball in orange_balls:
        x, y = ball["center"]
        r = ball["radius"]
        cv2.circle(display, (x, y), r, (0, 140, 255), 2)
        cv2.putText(
            display,
            f"orange ({x},{y})",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 140, 255),
            2
        )

    for ball in white_balls:
        x, y = ball["center"]
        r = ball["radius"]
        cv2.circle(display, (x, y), r, (255, 255, 255), 2)
        cv2.putText(
            display,
            f"white ({x},{y})",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    cv2.imshow("camera", display)
    cv2.imshow("red mask", mask_r)
    cv2.imshow("board mask", board_mask)
    cv2.imshow("play area mask", play_area_mask)
    cv2.imshow("orange mask", orange_mask)
    cv2.imshow("white mask", white_mask)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()