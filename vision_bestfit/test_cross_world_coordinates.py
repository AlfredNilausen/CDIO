import cv2
import numpy as np

from red_mask import red_mask
from boundary_points import extract_boundary_points
from best_fit_lines import fit_frame_lines
from geometry import draw_model_line, intersect_horizontal_vertical
from cross_detection import get_red_cross_mask, find_cross_candidate
from homography import (
    compute_homographies,
    pixel_to_world,
    world_to_view,
    draw_world_grid,
    BOARD_WIDTH_MM,
    BOARD_HEIGHT_MM
)

CAMERA_INDEX = 0
SMOOTH_ALPHA = 0.80
CROSS_SMOOTH_ALPHA = 0.70

prev_corners = None
prev_cross_center = None


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
            0.5,
            (255, 255, 255),
            1
        )


def draw_cross_info_camera(frame, cross_px, cross_world):
    """
    Tegner krydsets pixel- og world-koordinater i cameravinduet.
    """
    cx, cy = cross_px
    wx, wy = cross_world

    cv2.putText(
        frame,
        f"Cross px = ({cx}, {cy})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Cross mm = ({wx:.1f}, {wy:.1f})",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


def draw_cross_in_world(world_img, cross_world, display_scale=0.5):
    """
    Tegner korset i world-vinduet.
    """
    wx, wy = cross_world
    p = world_to_view((wx, wy), display_scale)

    cv2.circle(world_img, p, 8, (0, 255, 255), -1)
    cv2.putText(
        world_img,
        f"cross ({wx:.1f}, {wy:.1f}) mm",
        (p[0] + 10, p[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("red mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("cross search mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("cross mask", cv2.WINDOW_NORMAL)
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

    draw_model_line(display, models["top"], (255, 0, 0), 2)       # blå
    draw_model_line(display, models["bottom"], (0, 0, 255), 2)    # rød
    draw_model_line(display, models["left"], (0, 255, 0), 2)      # grøn
    draw_model_line(display, models["right"], (0, 255, 255), 2)   # gul

    corners = None
    cross_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    search_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    world_img = np.zeros((500, 700, 3), dtype=np.uint8)
    cv2.putText(
        world_img,
        "Waiting for corners / cross...",
        (40, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

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

            # ---------------------------------------------
            # 1) Homografi til world
            # ---------------------------------------------
            H_px_to_world, H_px_to_view, disp_w, disp_h = compute_homographies(
                corners,
                display_scale=0.5
            )

            world_img = cv2.warpPerspective(frame, H_px_to_view, (disp_w, disp_h))
            draw_world_grid(world_img, display_scale=0.5, step_mm=200)
            draw_world_corner_labels(world_img, display_scale=0.5)

            # ---------------------------------------------
            # 2) Find rødt kryds inde i banen
            # ---------------------------------------------
            cross_mask, search_mask, _ = get_red_cross_mask(
                frame,
                corners,
                margin_px=35
            )

            cross = find_cross_candidate(cross_mask, corners)

            if cross is not None:
                cnt = cross["contour"]
                new_cross_center = cross["center"]

                # smooth center lidt
                cross_center = smooth_point(
                    prev_cross_center,
                    new_cross_center,
                    CROSS_SMOOTH_ALPHA
                )
                prev_cross_center = cross_center

                cx, cy = cross_center

                # tegn i camera
                cv2.drawContours(display, [cnt], -1, (0, 255, 255), 3)
                cv2.circle(display, (cx, cy), 7, (255, 255, 255), -1)

                x, y, w, h = cross["bbox"]
                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 255), 2)

                # ---------------------------------------------
                # 3) Pixel -> world for krydset
                # ---------------------------------------------
                cross_world = pixel_to_world(cross_center, H_px_to_world)

                # tekst i camera-vinduet
                draw_cross_info_camera(display, cross_center, cross_world)

                # tegn i world-vinduet
                draw_cross_in_world(world_img, cross_world, display_scale=0.5)

    cv2.imshow("camera", display)
    cv2.imshow("red mask", mask_r)
    cv2.imshow("cross search mask", search_mask)
    cv2.imshow("cross mask", cross_mask)
    cv2.imshow("world", world_img)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    if key == ord("p"):
        print("----- STATUS -----")
        if corners is None:
            print("Corners not available")
        else:
            print("Corners:")
            for name in ["TL", "TR", "BL", "BR"]:
                print(f"{name} = {corners[name]}")

        if prev_cross_center is None:
            print("Cross center not available")
        else:
            print(f"Cross pixel = {prev_cross_center}")
            if corners is not None:
                H_px_to_world, _, _, _ = compute_homographies(corners, display_scale=0.5)
                cross_world = pixel_to_world(prev_cross_center, H_px_to_world)
                print(f"Cross world = ({cross_world}, {cross_world}) mm")
        print("------------------")

cap.release()
cv2.destroyAllWindows()