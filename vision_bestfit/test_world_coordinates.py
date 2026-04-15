import cv2
import numpy as np

from red_mask import red_mask
from boundary_points import extract_boundary_points
from best_fit_lines import fit_frame_lines
from geometry import draw_model_line, intersect_horizontal_vertical
from homography import (
    compute_homographies,
    pixel_to_world,
    draw_world_grid,
    world_to_view,
    BOARD_WIDTH_MM,
    BOARD_HEIGHT_MM
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


def draw_camera_corner_info(frame, corners, H_px_to_world):
    """
    Viser både pixel-koordinater og world-koordinater i mm i camera-vinduet.
    """
    y0 = 25
    dy = 25
    order = ["TL", "TR", "BL", "BR"]

    for i, name in enumerate(order):
        px = corners[name]
        world = pixel_to_world(px, H_px_to_world)

        text = (
            f"{name} px={px} "
            f"mm=({world[0]:.1f}, {world[1]:.1f})"
        )

        cv2.putText(
            frame,
            text,
            (20, y0 + i * dy),
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
cv2.namedWindow("world", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)

    boundary_points = extract_boundary_points(mask)
    models = fit_frame_lines(boundary_points)

    # -------- tegn best-fit linjer --------
    draw_model_line(display, models["top"], (255, 0, 0), 3)       # blå
    draw_model_line(display, models["bottom"], (0, 0, 255), 3)    # rød
    draw_model_line(display, models["left"], (0, 255, 0), 3)      # grøn
    draw_model_line(display, models["right"], (0, 255, 255), 3)   # gul

    corners = None

    # tom default world-visning hvis hjørner endnu ikke findes
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

    # -------- find hjørner --------
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

            # tegn hjørner i camerabilledet
            draw_corner_labels(display, corners)

            # -------- homografi --------
            H_px_to_world, H_px_to_view, disp_w, disp_h = compute_homographies(
                corners,
                display_scale=0.5
            )

            # warp til world-view
            world_img = cv2.warpPerspective(frame, H_px_to_view, (disp_w, disp_h))

            # grid + faste world-hjørner
            draw_world_grid(world_img, display_scale=0.5, step_mm=200)
            draw_world_corner_labels(world_img, display_scale=0.5)

            # tekst i camera-vinduet
            draw_camera_corner_info(display, corners, H_px_to_world)

    cv2.imshow("camera", display)
    cv2.imshow("red mask", mask)
    cv2.imshow("world", world_img)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    if key == ord("p") and corners is not None:
        H_px_to_world, _, _, _ = compute_homographies(corners, display_scale=0.5)

        print("----- CAMERA PIXELS -----")
        for name in ["TL", "TR", "BL", "BR"]:
            print(f"{name} = {corners[name]}")

        print("----- WORLD MM -----")
        for name in ["TL", "TR", "BL", "BR"]:
            world = pixel_to_world(corners[name], H_px_to_world)
            print(f"{name} = ({world[0]:.2f}, {world[1]:.2f})")

        print("----------------------")

cap.release()
cv2.destroyAllWindows()