import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

# =========================================================
# IMPORTS FRA DINE EKSISTERENDE FILER (BANE + WORLD)
# =========================================================
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

# =========================================================
# BOLD DETEKTOR (DIN SAMLEDE VERSION)
# =========================================================
@dataclass
class BallEdgesConfig:
    canny_t1: int = 110
    canny_t2: int = 300
    blur_k: int = 31
    blur_sigma: int = 1
    dilate_k: int = 3
    dilate_it: int = 2


@dataclass
class HoughConfig:
    dp: float = 1.2
    min_dist: int = 18
    param2: int = 18
    min_radius: int = 6
    max_radius: int = 11


@dataclass
class DetectorConfig:
    ball_edges: BallEdgesConfig = field(default_factory=BallEdgesConfig)
    hough: HoughConfig = field(default_factory=HoughConfig)

    inner_radius_ratio: float = 0.35
    orange_inner_ratio: float = 0.55

    white_saturation_max: float = 130.0
    white_ratio_threshold: float = 0.08

    white_s_max: int = 130
    white_local_gray_delta: int = 8
    white_local_v_delta: int = 8

    border_margin: int = 30
    border_white_saturation_max: float = 145.0
    border_white_ratio_threshold: float = 0.05

    orange_h_min: int = 12
    orange_h_max: int = 35
    orange_s_min: int = 70
    orange_v_min: int = 85

    orange_ratio_threshold: float = 0.05
    orange_strict_ratio_threshold: float = 0.12
    orange_saturation_min: float = 95.0


def ensure_odd(n: int):
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def dedupe_circles(circles: np.ndarray):
    kept = []
    for (cx, cy, r) in circles:
        duplicate = False
        for (kx, ky, kr) in kept:
            if (cx - kx) ** 2 + (cy - ky) ** 2 < 9 and abs(r - kr) <= 1:
                duplicate = True
                break
        if not duplicate:
            kept.append((cx, cy, r))
    return kept


def compute_ball_edges(frame, cfg):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (ensure_odd(cfg.blur_k), ensure_odd(cfg.blur_k)), cfg.blur_sigma)
    edges = cv2.Canny(gray, cfg.canny_t1, cfg.canny_t2)

    dk = ensure_odd(cfg.dilate_k)
    if dk > 1 and cfg.dilate_it > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
        edges = cv2.dilate(edges, kernel, iterations=cfg.dilate_it)

    return edges


def compute_white_mask(frame, cfg):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)
    return ((S < cfg.white_s_max) & (V > 120)).astype(np.uint8) * 255


def compute_orange_mask(frame, cfg):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    return (
        (H >= cfg.orange_h_min) &
        (H <= cfg.orange_h_max) &
        (S >= cfg.orange_s_min) &
        (V >= cfg.orange_v_min)
    ).astype(np.uint8) * 255


def detect_balls(frame, cfg):
    edges = compute_ball_edges(frame, cfg.ball_edges)
    white_mask = compute_white_mask(frame, cfg)
    orange_mask = compute_orange_mask(frame, cfg)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=cfg.hough.dp,
        minDist=cfg.hough.min_dist,
        param1=cfg.ball_edges.canny_t2,
        param2=cfg.hough.param2,
        minRadius=cfg.hough.min_radius,
        maxRadius=cfg.hough.max_radius,
    )

    white, orange = [], []

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        circles = dedupe_circles(circles)

        h, w = frame.shape[:2]

        for (cx, cy, r) in circles:
            x0 = max(0, cx - r)
            x1 = min(w, cx + r)
            y0 = max(0, cy - r)
            y1 = min(h, cy + r)

            if x1 <= x0 or y1 <= y0:
                continue

            w_ratio = np.mean(white_mask[y0:y1, x0:x1] > 0)
            o_ratio = np.mean(orange_mask[y0:y1, x0:x1] > 0)

            if o_ratio > w_ratio:
                orange.append((cx, cy, r))
            else:
                white.append((cx, cy, r))

    return white, orange, edges


def draw_balls(frame, whites, oranges):
    dbg = frame.copy()
    for (x, y, r) in whites:
        cv2.circle(dbg, (x, y), r, (0, 255, 255), 2)
        cv2.circle(dbg, (x, y), 2, (0, 255, 255), -1)

    for (x, y, r) in oranges:
        cv2.circle(dbg, (x, y), r, (0, 128, 255), 2)
        cv2.circle(dbg, (x, y), 2, (0, 128, 255), -1)

    return dbg


# =========================================================
# HJÆLPER TIL WORLD / HJØRNER
# =========================================================
CAMERA_INDEX = 0
SMOOTH_ALPHA = 0.80
DISPLAY_SCALE = 0.5

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
    y0 = 25
    dy = 25
    order = ["TL", "TR", "BL", "BR"]

    for i, name in enumerate(order):
        px = corners[name]
        world = pixel_to_world(px, H_px_to_world)

        text = f"{name} px={px} mm=({world[0]:.1f}, {world[1]:.1f})"

        cv2.putText(
            frame,
            text,
            (20, y0 + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


# =========================================================
# RØDT KRYDS I WORLD-VIEW
# =========================================================
def view_to_world(point_view, display_scale=0.5):
    """
    Konverterer fra world-view pixels tilbage til world mm.
    world_to_view bruger:
        x = x_mm * scale
        y = BOARD_HEIGHT_MM*scale - y_mm*scale
    så her inverterer vi.
    """
    x_view, y_view = point_view
    x_mm = float(x_view) / float(display_scale)
    y_mm = float(BOARD_HEIGHT_MM) - float(y_view) / float(display_scale)
    return x_mm, y_mm


def detect_red_cross_in_world(world_img_raw, display_scale=0.5):
    """
    Finder det røde kryds i det rectificerede world-view.
    Strategi:
      - rød maske i world_img
      - kun søg i central zone (så banen/rød kant ignoreres)
      - vælg største røde component i centerområdet
      - center = centroid
      - ender = venstre, højre, top, bund yderpunkt på konturen

    Returnerer dict eller None.
    """
    hsv = cv2.cvtColor(world_img_raw, cv2.COLOR_BGR2HSV)
    mask_red = red_mask(hsv)

    h, w = mask_red.shape

    # Søg kun i midterområdet for at undgå den røde kant
    x0 = int(0.25 * w)
    x1 = int(0.75 * w)
    y0 = int(0.25 * h)
    y1 = int(0.75 * h)

    search_mask = np.zeros_like(mask_red)
    search_mask[y0:y1, x0:x1] = mask_red[y0:y1, x0:x1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    search_mask = cv2.morphologyEx(search_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    search_mask = cv2.morphologyEx(search_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    # vælg største kontur i centerområdet
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area < 50:
        return None

    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-8:
        return None

    cx = int(round(M["m10"] / M["m00"]))
    cy = int(round(M["m01"] / M["m00"]))
    center = (cx, cy)

    pts = cnt[:, 0, :]  # shape (N,2)

    left_end = tuple(pts[np.argmin(pts[:, 0])])
    right_end = tuple(pts[np.argmax(pts[:, 0])])
    top_end = tuple(pts[np.argmin(pts[:, 1])])
    bottom_end = tuple(pts[np.argmax(pts[:, 1])])

    return {
        "mask": search_mask,
        "contour": cnt,
        "area": area,
        "center_view": center,
        "left_view": left_end,
        "right_view": right_end,
        "top_view": top_end,
        "bottom_view": bottom_end,
        "center_mm": view_to_world(center, display_scale),
        "left_mm": view_to_world(left_end, display_scale),
        "right_mm": view_to_world(right_end, display_scale),
        "top_mm": view_to_world(top_end, display_scale),
        "bottom_mm": view_to_world(bottom_end, display_scale),
    }


def draw_world_point_with_label(world_img, point_view, point_mm, label, color):
    px, py = point_view
    mmx, mmy = point_mm

    cv2.circle(world_img, (int(px), int(py)), 6, color, -1)
    cv2.putText(
        world_img,
        f"{label} ({int(mmx)}, {int(mmy)})",
        (int(px) + 8, int(py) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1
    )


def draw_cross_info_in_world(world_img, cross_info):
    if cross_info is None:
        return

    cnt = cross_info["contour"]
    cv2.drawContours(world_img, [cnt], -1, (255, 255, 255), 2)

    draw_world_point_with_label(
        world_img,
        cross_info["center_view"],
        cross_info["center_mm"],
        "C",
        (255, 255, 255)
    )
    draw_world_point_with_label(
        world_img,
        cross_info["left_view"],
        cross_info["left_mm"],
        "L",
        (255, 0, 255)
    )
    draw_world_point_with_label(
        world_img,
        cross_info["right_view"],
        cross_info["right_mm"],
        "R",
        (0, 255, 255)
    )
    draw_world_point_with_label(
        world_img,
        cross_info["top_view"],
        cross_info["top_mm"],
        "T",
        (0, 255, 0)
    )
    draw_world_point_with_label(
        world_img,
        cross_info["bottom_view"],
        cross_info["bottom_mm"],
        "B",
        (0, 165, 255)
    )


# =========================================================
# MAIN
# =========================================================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    raise SystemExit

cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("red mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("world", cv2.WINDOW_NORMAL)
cv2.namedWindow("ball edges", cv2.WINDOW_NORMAL)
cv2.namedWindow("cross mask", cv2.WINDOW_NORMAL)

cfg = DetectorConfig()

last_cross_info = None
last_corners = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)

    boundary_points = extract_boundary_points(mask)
    models = fit_frame_lines(boundary_points)

    # -------- tegn bane-linjer --------
    draw_model_line(display, models["top"], (255, 0, 0), 3)
    draw_model_line(display, models["bottom"], (0, 0, 255), 3)
    draw_model_line(display, models["left"], (0, 255, 0), 3)
    draw_model_line(display, models["right"], (0, 255, 255), 3)

    # default world img
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

    cross_mask_vis = np.zeros((500, 700), dtype=np.uint8)

    H_px_to_world = None
    H_px_to_view = None
    corners = None
    cross_info = None

    # =========================================================
    # 1) FIND HJØRNER OG HOMOGRAFI
    # =========================================================
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
            last_corners = corners

            draw_corner_labels(display, corners)

            H_px_to_world, H_px_to_view, disp_w, disp_h = compute_homographies(
                corners,
                display_scale=DISPLAY_SCALE
            )

            # warp til world-view
            world_img_raw = cv2.warpPerspective(frame, H_px_to_view, (disp_w, disp_h))
            world_img = world_img_raw.copy()

            # grid + world labels
            draw_world_grid(world_img, display_scale=DISPLAY_SCALE, step_mm=200)
            draw_world_corner_labels(world_img, display_scale=DISPLAY_SCALE)

            # tekst på camerabilledet
            draw_camera_corner_info(display, corners, H_px_to_world)

            # =========================================================
            # 2) FIND RØDT KRYDS I WORLD
            # =========================================================
            cross_info = detect_red_cross_in_world(world_img_raw, display_scale=DISPLAY_SCALE)
            last_cross_info = cross_info

            if cross_info is not None:
                draw_cross_info_in_world(world_img, cross_info)
                cross_mask_vis = cross_info["mask"]

    # =========================================================
    # 3) FIND BOLDE
    # =========================================================
    whites, oranges, edges = detect_balls(frame, cfg)
    display = draw_balls(display, whites, oranges)

    # =========================================================
    # 4) VIS BOLDE I WORLD MM
    # =========================================================
    if H_px_to_world is not None:
        for (x, y, r) in whites:
            wcoord = pixel_to_world((x, y), H_px_to_world)
            p = world_to_view(wcoord, DISPLAY_SCALE)

            cv2.circle(world_img, p, 6, (0, 255, 255), -1)
            cv2.putText(
                world_img,
                f"W ({int(wcoord[0])}, {int(wcoord[1])})",
                (p[0] + 8, p[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1
            )

        for (x, y, r) in oranges:
            wcoord = pixel_to_world((x, y), H_px_to_world)
            p = world_to_view(wcoord, DISPLAY_SCALE)

            cv2.circle(world_img, p, 6, (0, 128, 255), -1)
            cv2.putText(
                world_img,
                f"O ({int(wcoord[0])}, {int(wcoord[1])})",
                (p[0] + 8, p[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 128, 255),
                1
            )

    # =========================================================
    # VIS
    # =========================================================
    cv2.imshow("camera", display)
    cv2.imshow("red mask", mask)
    cv2.imshow("world", world_img)
    cv2.imshow("ball edges", edges)

    # vis cross mask kun hvis størrelse matcher; ellers resize til preview
    if cross_mask_vis is not None and cross_mask_vis.size > 0:
        cv2.imshow("cross mask", cross_mask_vis)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    if key == ord("p"):
        print("\n================ SNAPSHOT ================")

        if last_corners is not None:
            H_px_to_world_print, _, _, _ = compute_homographies(last_corners, display_scale=DISPLAY_SCALE)

            print("----- CAMERA PIXELS (corners) -----")
            for name in ["TL", "TR", "BL", "BR"]:
                print(f"{name} = {last_corners[name]}")

            print("----- WORLD MM (corners) -----")
            for name in ["TL", "TR", "BL", "BR"]:
                world = pixel_to_world(last_corners[name], H_px_to_world_print)
                print(f"{name} = ({world[0]:.2f}, {world[1]:.2f})")

        if last_cross_info is not None:
            print("----- RED CROSS (WORLD MM) -----")
            print(f"CENTER = ({last_cross_info['center_mm'][0]:.2f}, {last_cross_info['center_mm'][1]:.2f})")
            print(f"LEFT   = ({last_cross_info['left_mm'][0]:.2f}, {last_cross_info['left_mm'][1]:.2f})")
            print(f"RIGHT  = ({last_cross_info['right_mm'][0]:.2f}, {last_cross_info['right_mm'][1]:.2f})")
            print(f"TOP    = ({last_cross_info['top_mm'][0]:.2f}, {last_cross_info['top_mm'][1]:.2f})")
            print(f"BOTTOM = ({last_cross_info['bottom_mm'][0]:.2f}, {last_cross_info['bottom_mm'][1]:.2f})")
        else:
            print("RED CROSS: not found")

        print("------------------------------------------")

cap.release()
cv2.destroyAllWindows()
