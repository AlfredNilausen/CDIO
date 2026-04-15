import cv2
import numpy as np

from red_mask import red_mask
from boundary_points import extract_boundary_points
from best_fit_lines import fit_frame_lines
from geometry import draw_model_line, intersect_horizontal_vertical
from homography import (
    compute_homographies,
    pixel_to_world,
    world_to_view,
    draw_world_grid,
    BOARD_WIDTH_MM,
    BOARD_HEIGHT_MM
)
from ball_detection import (
    make_play_area_mask,
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
            name,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


def draw_world_corner_labels(world_img, display_scale=0.5):
    labels = {
        "BL": (0, 0),
        "TL": (0, BOARD_HEIGHT_MM),
        "TR": (BOARD_WIDTH_MM, BOARD_HEIGHT_MM),
        "BR": (BOARD_WIDTH_MM, 0),
    }

    for name, pt_mm in labels.items():
        p = world_to_view(pt_mm, display_scale)
        cv2.circle(world_img, p, 6, (0, 255, 0), -1)
        cv2.putText(
            world_img,
            f"{name} {pt_mm}",
            (p[0] + 8, p[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )


def draw_ball_camera(frame, ball, label, color_bgr, H_px_to_world, line_y):
    cx, cy = ball["center"]
    radius = ball["radius"]

    wx, wy = pixel_to_world((cx, cy), H_px_to_world)

    cv2.circle(frame, (cx, cy), radius, color_bgr, 2)
    cv2.putText(
        frame,
        f"{label} px=({cx},{cy})",
        (20, line_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_bgr,
        2
    )
    cv2.putText(
        frame,
        f"{label} mm=({wx:.1f},{wy:.1f})",
        (20, line_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_bgr,
        2
    )


def draw_ball_world(world_img, ball, H_px_to_world, display_scale, label, color_bgr):
    cx, cy = ball["center"]
    wx, wy = pixel_to_world((cx, cy), H_px_to_world)
    p = world_to_view((wx, wy), display_scale)

    cv2.circle(world_img, p, 7, color_bgr, -1)
    cv2.putText(
        world_img,
        f"{label} ({wx:.1f},{wy:.1f})",
        (p[0] + 8, p[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color_bgr,
        1
    )


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("red mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("play area mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("orange mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("white mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("world", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_r = red_mask(hsv)

    boundary_points = extract_boundary_points(mask_r)
    models = fit_frame_lines(boundary_points)

    draw_model_line(display, models["top"], (255, 0, 0), 2)
    draw_model_line(display, models["bottom"], (0, 0, 255), 2)
    draw_model_line(display, models["left"], (0, 255, 0), 2)
    draw_model_line(display, models["right"], (0, 255, 255), 2)

    corners = None
    play_area_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    orange_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    white_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    world_img = np.zeros((500, 700, 3), dtype=np.uint8)
    cv2.putText(
        world_img,
        "Waiting for stable corners...",
        (40, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    orange_balls = []
    white_balls = []

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

            # Homografi
            H_px_to_world, H_px_to_view, disp_w, disp_h = compute_homographies(
                corners,
                display_scale=0.5
            )

            world_img = cv2.warpPerspective(frame, H_px_to_view, (disp_w, disp_h))
            draw_world_grid(world_img, display_scale=0.5, step_mm=200)
            draw_world_corner_labels(world_img, display_scale=0.5)

            # Søgeområde for bolde
            play_area_mask = make_play_area_mask(
                frame.shape,
                corners,
                mask_r,
                margin_px=18
            )

            # Masker
            orange_mask = get_orange_ball_mask(frame, roi_mask=play_area_mask)
            white_mask = get_white_ball_mask(frame, roi_mask=play_area_mask)

            # Kandidater
            orange_balls = find_ball_candidates(
                orange_mask,
                min_area=20,
                max_area=1200,
                min_circularity=0.45
            )

            white_balls = find_ball_candidates(
                white_mask,
                min_area=20,
                max_area=1200,
                min_circularity=0.45
            )

            # Tegn orange bolde
            base_y = 100
            for i, ball in enumerate(orange_balls):
                draw_ball_camera(
                    display,
                    ball,
                    f"orange{i+1}",
                    (0, 140, 255),
                    H_px_to_world,
                    base_y + i * 50
                )
                draw_ball_world(
                    world_img,
                    ball,
                    H_px_to_world,
                    0.5,
                    f"orange{i+1}",
                    (0, 140, 255)
                )

            # Tegn hvide bolde
            base_y = 260
            for i, ball in enumerate(white_balls):
                draw_ball_camera(
                    display,
                    ball,
                    f"white{i+1}",
                    (255, 255, 255),
                    H_px_to_world,
                    base_y + i * 50
                )
                draw_ball_world(
                    world_img,
                    ball,
                    H_px_to_world,
                    0.5,
                    f"white{i+1}",
                    (255, 255, 255)
                )

    cv2.imshow("camera", display)
    cv2.imshow("red mask", mask_r)
    cv2.imshow("play area mask", play_area_mask)
    cv2.imshow("orange mask", orange_mask)
    cv2.imshow("white mask", white_mask)
    cv2.imshow("world", world_img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    if key == ord("p"):
        print("----- BALL STATUS -----")
        print(f"Orange balls: {len(orange_balls)}")
        for i, ball in enumerate(orange_balls, start=1):
            print(f"  orange{i}: px={ball['center']}")

        print(f"White balls: {len(white_balls)}")
        for i, ball in enumerate(white_balls, start=1):
            print(f"  white{i}: px={ball['center']}")
        print("-----------------------")

cap.release()
cv2.destroyAllWindows()