import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# =========================================================
# CONFIG
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

    # ----- WHITE klassifikation -----
    # lidt strammere så orange ikke så let bliver accepteret som white
    white_saturation_max: float = 130.0
    white_ratio_threshold: float = 0.08

    # White mask (adaptiv)
    white_s_max: int = 130
    white_local_gray_delta: int = 8
    white_local_v_delta: int = 8

    # Border rescue (white)
    border_margin: int = 30
    border_white_saturation_max: float = 145.0
    border_white_ratio_threshold: float = 0.05

    # ----- ORANGE klassifikation -----
    # bredere orange-område så gul-orange også fanges
    # ----- ORANGE mask -----
    # målrettet gul-orange bold og mindre mod den røde kant
    orange_h_min: int = 12
    orange_h_max: int = 35
    orange_s_min: int = 70
    orange_v_min: int = 85

    # LAB: orange/gul har høj b-kanal
    orange_lab_a_min: int = 118
    orange_lab_b_min: int = 145

    # connected components filter
    orange_area_min: int = 8
    orange_area_max: int = 220

    # ratio thresholds
    orange_ratio_threshold: float = 0.05
    orange_strict_ratio_threshold: float = 0.12
    orange_saturation_min: float = 95.0



# =========================================================
# HELPERS
# =========================================================
def ensure_odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def dedupe_circles(circles: np.ndarray) -> List[Tuple[int, int, int]]:
    kept = []
    for (cx, cy, r) in circles:
        duplicate = False
        for (kx, ky, kr) in kept:
            if (cx - kx) ** 2 + (cy - ky) ** 2 < 9 and abs(r - kr) <= 1:
                duplicate = True
                break
        if not duplicate:
            kept.append((int(cx), int(cy), int(r)))
    return kept


# =========================================================
# EDGE DETECTION
# =========================================================
def compute_ball_edges(frame: np.ndarray, cfg: BallEdgesConfig) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    k = ensure_odd(cfg.blur_k)
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), cfg.blur_sigma)

    edges = cv2.Canny(gray, cfg.canny_t1, cfg.canny_t2)

    dk = ensure_odd(cfg.dilate_k)
    if dk > 1 and cfg.dilate_it > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
        edges = cv2.dilate(edges, kernel, iterations=cfg.dilate_it)

    return edges


# =========================================================
# MASKS
# =========================================================
def compute_white_mask(frame: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    H, S, V = cv2.split(hsv)

    # Lokal baggrundsmodel
    bg_gray = cv2.GaussianBlur(gray, (41, 41), 0)
    bg_v = cv2.GaussianBlur(V, (41, 41), 0)

    # Adaptiv lyshed: lysere end lokal baggrund
    mask_local_gray = gray > (bg_gray + cfg.white_local_gray_delta)
    mask_local_v = V > (bg_v + cfg.white_local_v_delta)

    # White bør have lav/moderat saturation
    mask_low_sat = S < cfg.white_s_max

    # Ekskludér orange ud fra de felter der faktisk findes i config nu
    orange_like = (
        (H >= cfg.orange_h_min) &
        (H <= cfg.orange_h_max) &
        (S >= 70) &
        (V >= 70)
    )

    white_mask = (
        mask_low_sat &
        (mask_local_gray | mask_local_v) &
        (~orange_like)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return white_mask


def compute_orange_mask(frame: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Strammere orange/gul-orange maske
    mask_hsv = (
        (H >= cfg.orange_h_min) &
        (H <= cfg.orange_h_max) &
        (S >= cfg.orange_s_min) &
        (V >= cfg.orange_v_min)
    )

    # Ekskludér meget rød kant
    exclude_red = (
        (H < 10) &
        (S > 90) &
        (V > 70)
    )

    orange_mask = (mask_hsv & (~exclude_red)).astype(np.uint8) * 255

    # Bevar små orange områder, men undgå for aggressiv udvidelse
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return orange_mask



# =========================================================
# ROI / FEATURES
# =========================================================
def get_circle_roi(frame, cx: int, cy: int, r: int, ratio: float):
    h, w = frame.shape[:2]

    x0, y0 = max(0, cx - r), max(0, cy - r)
    x1, y1 = min(w, cx + r + 1), min(h, cy + r + 1)

    roi = frame[y0:y1, x0:x1]

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    inner_r = max(2, min(int(r * ratio), max(2, r - 3)))
    cv2.circle(mask, (cx - x0, cy - y0), inner_r, 255, -1)

    return roi, mask


def circle_brightness(frame, cx: int, cy: int, r: int, ratio: float) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi, mask = get_circle_roi(gray, cx, cy, r, ratio)
    values = roi[mask > 0]
    return float(np.mean(values)) if len(values) > 0 else 0.0


def circle_saturation(frame, cx: int, cy: int, r: int, ratio: float) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)
    roi, mask = get_circle_roi(S, cx, cy, r, ratio)
    values = roi[mask > 0]
    return float(np.mean(values)) if len(values) > 0 else 999.0


def ring_contrast(
    frame,
    cx: int,
    cy: int,
    r: int,
    inner_ratio: float = 0.45,
    outer_ratio: float = 1.6
) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    y_grid, x_grid = np.ogrid[:h, :w]
    dist2 = (x_grid - cx) ** 2 + (y_grid - cy) ** 2

    inner_r = max(2, int(r * inner_ratio))
    outer_r = max(inner_r + 1, int(r * outer_ratio))

    inner_mask = dist2 <= inner_r ** 2
    ring_mask = (dist2 > r ** 2) & (dist2 <= outer_r ** 2)

    inner_vals = gray[inner_mask]
    ring_vals = gray[ring_mask]

    if inner_vals.size == 0 or ring_vals.size == 0:
        return 0.0

    return float(np.mean(inner_vals) - np.mean(ring_vals))


def circle_mask_ratio(mask_img: np.ndarray, cx: int, cy: int, r: int, ratio: float) -> float:
    roi, mask = get_circle_roi(mask_img, cx, cy, r, ratio)
    valid = mask > 0
    if np.count_nonzero(valid) == 0:
        return 0.0
    pos_pixels = np.count_nonzero((roi > 0) & valid)
    total_pixels = np.count_nonzero(valid)
    return float(pos_pixels) / float(total_pixels)


# =========================================================
# CLASSIFICATION
# =========================================================
def classify_ball(frame, white_mask, orange_mask, cx: int, cy: int, r: int, cfg: DetectorConfig):
    h, w = frame.shape[:2]

    brightness = circle_brightness(frame, cx, cy, r, cfg.inner_radius_ratio)
    saturation = circle_saturation(frame, cx, cy, r, cfg.inner_radius_ratio)
    contrast = ring_contrast(frame, cx, cy, r, inner_ratio=0.45, outer_ratio=1.6)

    white_ratio = circle_mask_ratio(white_mask, cx, cy, r, cfg.inner_radius_ratio)
    orange_ratio = circle_mask_ratio(orange_mask, cx, cy, r, cfg.orange_inner_ratio)

    dist_to_border = min(cx, cy, w - cx, h - cy)

    label = "UNKNOWN"

    # ----------------------------
    # WHITE candidate
    # ----------------------------
    white_ok = False
    if (
        saturation < cfg.white_saturation_max
        and white_ratio > cfg.white_ratio_threshold
        and white_ratio >= orange_ratio
    ):
        white_ok = True

    if (
        saturation < 120
        and white_ratio > 0.04
        and orange_ratio < 0.05
        and brightness > 120
        and contrast > 3.0
    ):
        white_ok = True

    if dist_to_border < cfg.border_margin:
        if (
            saturation < cfg.border_white_saturation_max
            and white_ratio > cfg.border_white_ratio_threshold
            and white_ratio >= orange_ratio
        ):
            white_ok = True

    # ----------------------------
    # ORANGE candidate
    # ----------------------------
    orange_ok = False

    if (
            orange_ratio > cfg.orange_strict_ratio_threshold
            and saturation > 95
    ):
        orange_ok = True

    if (
            orange_ratio > cfg.orange_ratio_threshold
            and saturation > cfg.orange_saturation_min
            and orange_ratio >= white_ratio
    ):
        orange_ok = True
    # ----------------------------
    # ----------------------------
    # FINAL DECISION
    # ----------------------------
    if orange_ok and (orange_ratio > white_ratio or saturation > 140):
        label = "ORANGE"
    elif white_ok and white_ratio >= orange_ratio:
        label = "WHITE"
    elif orange_ok:
        label = "ORANGE"
    elif white_ok:
        label = "WHITE"

    features = {
        "brightness": brightness,
        "saturation": saturation,
        "contrast": contrast,
        "white_ratio": white_ratio,
        "orange_ratio": orange_ratio,
        "dist_to_border": dist_to_border,
    }

    return label, features



# =========================================================
# DETECTION
# =========================================================
def detect_balls(frame, cfg: DetectorConfig):
    edges = compute_ball_edges(frame, cfg.ball_edges)
    white_mask = compute_white_mask(frame, cfg)
    orange_mask = compute_orange_mask(frame, cfg)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = ensure_odd(cfg.ball_edges.blur_k)
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), cfg.ball_edges.blur_sigma)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=cfg.hough.dp,
        minDist=cfg.hough.min_dist,
        param1=cfg.ball_edges.canny_t2,
        param2=cfg.hough.param2,
        minRadius=cfg.hough.min_radius,
        maxRadius=cfg.hough.max_radius,
    )

    white_balls = []
    orange_balls = []
    unknown_balls = []

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        kept = dedupe_circles(circles)

        for (cx, cy, r) in kept:
            label, feat = classify_ball(frame, white_mask, orange_mask, cx, cy, r, cfg)

            print(
                f"Ball ({cx},{cy}) {label} "
                f"B={feat['brightness']:.1f} "
                f"S={feat['saturation']:.1f} "
                f"C={feat['contrast']:.1f} "
                f"W={feat['white_ratio']:.2f} "
                f"O={feat['orange_ratio']:.2f} "
                f"D={feat['dist_to_border']}"
            )

            item = (cx, cy, r, feat)
            if label == "WHITE":
                white_balls.append(item)
            elif label == "ORANGE":
                orange_balls.append(item)
            else:
                unknown_balls.append(item)

    return white_balls, orange_balls, unknown_balls, edges, white_mask, orange_mask


# =========================================================
# DRAW / DEBUG
# =========================================================
def draw_detections(frame, white_balls, orange_balls, unknown_balls):
    dbg = frame.copy()

    # White = yellow
    for (cx, cy, r, feat) in white_balls:
        cv2.circle(dbg, (cx, cy), r, (0, 255, 255), 2)
        cv2.circle(dbg, (cx, cy), 2, (0, 255, 255), -1)
        txt = f"W B{feat['brightness']:.0f} S{feat['saturation']:.0f} C{feat['contrast']:.1f}"
        cv2.putText(dbg, txt, (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

    # Orange = orange-ish / red
    for (cx, cy, r, feat) in orange_balls:
        cv2.circle(dbg, (cx, cy), r, (0, 128, 255), 2)
        cv2.circle(dbg, (cx, cy), 2, (0, 128, 255), -1)
        txt = f"O B{feat['brightness']:.0f} S{feat['saturation']:.0f} O{feat['orange_ratio']:.2f}"
        cv2.putText(dbg, txt, (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 128, 255), 1)

    # Unknown = red
    for (cx, cy, r, feat) in unknown_balls:
        cv2.circle(dbg, (cx, cy), r, (0, 0, 255), 1)
        cv2.putText(dbg, "?", (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1)

    cv2.putText(dbg, f"White balls: {len(white_balls)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(dbg, f"Orange balls: {len(orange_balls)}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
    cv2.putText(dbg, f"Unknown: {len(unknown_balls)}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return dbg


def make_mask_overlay(frame, mask, color=(0, 255, 0), alpha=0.25):
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_vis[mask > 0] = color
    overlay = cv2.addWeighted(frame, 1.0 - alpha, mask_vis, alpha, 0)
    return overlay


# =========================================================
# MAIN
# =========================================================
def run_camera_detector():
    cfg = DetectorConfig()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("white_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("orange_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("white_overlay", cv2.WINDOW_NORMAL)
    cv2.namedWindow("orange_overlay", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        white_balls, orange_balls, unknown_balls, edges, white_mask, orange_mask = detect_balls(frame, cfg)

        dbg = draw_detections(frame, white_balls, orange_balls, unknown_balls)
        white_overlay = make_mask_overlay(frame, white_mask, color=(0, 255, 0), alpha=0.25)
        orange_overlay = make_mask_overlay(frame, orange_mask, color=(0, 128, 255), alpha=0.30)

        cv2.imshow("camera", dbg)
        cv2.imshow("edges", edges)
        cv2.imshow("white_mask", white_mask)
        cv2.imshow("orange_mask", orange_mask)
        cv2.imshow("white_overlay", white_overlay)
        cv2.imshow("orange_overlay", orange_overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("s"):
            print("\n=== CURRENT DETECTIONS ===")
            print("WHITE:", [(cx, cy, r) for (cx, cy, r, _) in white_balls])
            print("ORANGE:", [(cx, cy, r) for (cx, cy, r, _) in orange_balls])
            print("UNKNOWN:", [(cx, cy, r) for (cx, cy, r, _) in unknown_balls])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_detector()