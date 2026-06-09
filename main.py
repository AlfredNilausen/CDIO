import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_bestfit"))

"""
main.py  —  Integreret vision + rute + retning

Tastetryk:
  r  —  genberegn rute fra nuværende detektion
  p  —  print snapshot til terminal
  ESC — afslut
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import math
import time

# ── Vision-moduler (fra samlet.py) ──────────────────────────────────────────
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
    BOARD_HEIGHT_MM,
)

# ════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX = 0
SMOOTH_ALPHA = 0.80
DISPLAY_SCALE = 0.5

# ArUco
ARUCO_DICT = aruco.DICT_4X4_50
ROBOT_MARKER_ID = 0
MARKER_SIZE_MM = 80  # fysisk størrelse af printet marker i mm

# Rute
CENTER_EXCLUSION_MM = 150  # eksklusionszone rundt om krydset (mm)
GATE_OFFSET_MM = 320  # indgangspunkt afstand fra krydset (mm) - større = robotten svinger mere udenom
WALL_MARGIN_MM = 180  # bold inden for dette fra kanten behandles som kantbold
WALL_APPROACH_MM = 250  # start tilkørsel denne afstand fra bolden (vinkelret på kant)

# ════════════════════════════════════════════════════════════════════════════
# BOLD-DETEKTION  (fra samlet.py)
# ════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field


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
    white_s_max: int = 130
    orange_h_min: int = 12
    orange_h_max: int = 35
    orange_s_min: int = 70
    orange_v_min: int = 85


def _ensure_odd(n):
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def _dedupe(circles):
    kept = []
    for (cx, cy, r) in circles:
        if not any((cx - kx) ** 2 + (cy - ky) ** 2 < 9 and abs(r - kr) <= 1
                   for kx, ky, kr in kept):
            kept.append((cx, cy, r))
    return kept


def detect_balls(frame, cfg: DetectorConfig):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,
                            (_ensure_odd(cfg.ball_edges.blur_k),) * 2,
                            cfg.ball_edges.blur_sigma)
    edges = cv2.Canny(gray, cfg.ball_edges.canny_t1, cfg.ball_edges.canny_t2)
    if cfg.ball_edges.dilate_it > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (_ensure_odd(cfg.ball_edges.dilate_k),) * 2)
        edges = cv2.dilate(edges, k, iterations=cfg.ball_edges.dilate_it)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    white_mask = ((S < cfg.white_s_max) & (V > 120)).astype(np.uint8) * 255
    orange_mask = ((H >= cfg.orange_h_min) & (H <= cfg.orange_h_max) &
                   (S >= cfg.orange_s_min) & (V >= cfg.orange_v_min)
                   ).astype(np.uint8) * 255

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=cfg.hough.dp, minDist=cfg.hough.min_dist,
        param1=cfg.ball_edges.canny_t2, param2=cfg.hough.param2,
        minRadius=cfg.hough.min_radius, maxRadius=cfg.hough.max_radius,
    )

    whites, oranges = [], []
    if circles is not None:
        h, w = frame.shape[:2]
        for (cx, cy, r) in _dedupe(np.round(circles[0]).astype(int)):
            x0, x1 = max(0, cx - r), min(w, cx + r)
            y0, y1 = max(0, cy - r), min(h, cy + r)
            if x1 <= x0 or y1 <= y0:
                continue
            wo = np.mean(white_mask[y0:y1, x0:x1] > 0)
            oo = np.mean(orange_mask[y0:y1, x0:x1] > 0)
            (oranges if oo > wo else whites).append((cx, cy, r))
    return whites, oranges, edges


# ════════════════════════════════════════════════════════════════════════════
# KRYDS-DETEKTION  (fra samlet.py)
# ════════════════════════════════════════════════════════════════════════════

def _view_to_world(pt_view, scale=DISPLAY_SCALE):
    x_mm = pt_view[0] / scale
    y_mm = BOARD_HEIGHT_MM - pt_view[1] / scale
    return x_mm, y_mm


def detect_red_cross_world(world_img_raw, scale=DISPLAY_SCALE):
    hsv = cv2.cvtColor(world_img_raw, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)
    h, w = mask.shape
    x0, x1 = int(.25 * w), int(.75 * w)
    y0, y1 = int(.25 * h), int(.75 * h)
    sm = np.zeros_like(mask)
    sm[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sm = cv2.morphologyEx(sm, cv2.MORPH_OPEN, k)
    sm = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return None
    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-8:
        return None
    cx = int(round(M["m10"] / M["m00"]))
    cy = int(round(M["m01"] / M["m00"]))
    pts = cnt[:, 0, :]
    return {
        "contour": cnt,
        "center_view": (cx, cy),
        "center_mm": _view_to_world((cx, cy), scale),
        "mask": sm,
    }


# ════════════════════════════════════════════════════════════════════════════
# RETNINGS-DETEKTION  (fra direction.py)
# ════════════════════════════════════════════════════════════════════════════

def _make_camera_matrix(w, h):
    f = max(w, h)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def _rvec_to_heading(rvec):
    R, _ = cv2.Rodrigues(rvec)
    dx, dy = R[0, 0], R[1, 0]
    return math.degrees(math.atan2(-dy, dx))


def _heading_cardinal(a):
    a = a % 360
    for thresh, label in [(22, "E"), (67, "NE"), (112, "N"), (157, "NW"),
                          (202, "W"), (247, "SW"), (292, "S"), (337, "SE"), (360, "E")]:
        if a < thresh:
            return label
    return "E"


_aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
_aruco_params = aruco.DetectorParameters()
_aruco_detector = aruco.ArucoDetector(_aruco_dict, _aruco_params)

_half = MARKER_SIZE_MM / 2.0
_OBJ_PTS = np.array([[-_half, _half, 0], [_half, _half, 0],
                     [_half, -_half, 0], [-_half, -_half, 0]], dtype=np.float32)


def detect_direction(frame):
    """Returnerer {'found', 'heading', 'cardinal', 'center_px', 'center_mm_fn'}"""
    H, W = frame.shape[:2]
    cam = _make_camera_matrix(W, H)
    dist = np.zeros((5, 1), dtype=np.float64)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _aruco_detector.detectMarkers(gray)
    if ids is None:
        return {"found": False}
    for i, mid in enumerate(ids.flatten()):
        if mid != ROBOT_MARKER_ID:
            continue
        mc = corners[i][0]
        ok, rvec, tvec = cv2.solvePnP(_OBJ_PTS, mc.astype(np.float32),
                                      cam, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        heading = _rvec_to_heading(rvec)
        cx = int(mc[:, 0].mean())
        cy = int(mc[:, 1].mean())
        return {
            "found": True,
            "heading": heading,
            "cardinal": _heading_cardinal(heading),
            "center_px": (cx, cy),
            "rvec": rvec,
            "tvec": tvec,
            "cam_mat": cam,
            "dist": dist,
        }
    return {"found": False}


# ════════════════════════════════════════════════════════════════════════════
# RUTE-PLANLÆGNING  (fra first_route.py — tilpasset til mm-koordinater)
# ════════════════════════════════════════════════════════════════════════════

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seg_dist(p1, p2, pt):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if dx == dy == 0:
        return _dist(p1, pt)
    t = max(0, min(1, ((pt[0] - p1[0]) * dx + (pt[1] - p1[1]) * dy) / (dx * dx + dy * dy)))
    return _dist((p1[0] + t * dx, p1[1] + t * dy), pt)


def _quadrant(pt, cross_mm):
    x = pt[0] - cross_mm[0]
    y = pt[1] - cross_mm[1]
    if x < 0 and y > 0: return 1  # top-left  (mm: x<cx, y>cy)
    if x > 0 and y > 0: return 2  # top-right
    if x > 0 and y < 0: return 3  # bottom-right
    return 4  # bottom-left


def _nearest_neighbor(start, pts):
    rem, path, cur = list(pts), [], start
    while rem:
        nxt = min(rem, key=lambda p: _dist(cur, p))
        path.append(nxt);
        cur = nxt;
        rem.remove(nxt)
    return path


def _approach_exit(target, cross_mm):
    if _dist(target, cross_mm) > CENTER_EXCLUSION_MM * 1.5:
        return [], []
    vx = target[0] - cross_mm[0];
    vy = target[1] - cross_mm[1]
    mag = math.hypot(vx, vy) or 1
    nx, ny = vx / mag, vy / mag
    d = CENTER_EXCLUSION_MM * 1.3
    wp = (target[0] + nx * d, target[1] + ny * d)
    return [wp], [wp]


def _detour(p1, p2, cross_mm):
    if _seg_dist(p1, p2, cross_mm) >= CENTER_EXCLUSION_MM:
        return [p2]

    # Vektor for linjen
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    mag_line = math.hypot(dx, dy)
    if mag_line == 0:
        return [p2]

    # Find det punkt på linjen, der er tættest på krydsets centrum
    t = max(0, min(1, ((cross_mm[0] - p1[0]) * dx + (cross_mm[1] - p1[1]) * dy) / (mag_line ** 2)))
    closest_x = p1[0] + t * dx
    closest_y = p1[1] + t * dy

    # Vektor fra krydset ud mod det tætteste punkt
    vx, vy = closest_x - cross_mm[0], closest_y - cross_mm[1]
    mag_v = math.hypot(vx, vy)

    # FIX: Hvis ruten går PRÆCIS gennem midten (vektorlængden er næsten 0),
    # bruger vi linjens tværvektor i stedet, så vi tvinger den ud til siden.
    if mag_v < 10:
        vx, vy = -dy, dx
        mag_v = math.hypot(vx, vy)

    # Skub waypointet ud fra centrum med en sikkerhedsmargin
    push = CENTER_EXCLUSION_MM * 1.4
    wp = (cross_mm[0] + (vx / mag_v) * push, cross_mm[1] + (vy / mag_v) * push)

    return [wp, p2]


def _entry_point_for_quadrant(target_q, cross_mm):
    """Indgangspunkt til nyt kvadrant, GATE_OFFSET_MM fra krydset."""
    cx, cy = cross_mm
    d = GATE_OFFSET_MM
    return {1: (cx - d, cy + d), 2: (cx + d, cy + d), 3: (cx + d, cy - d), 4: (cx - d, cy - d)}[target_q]


def _wall_approach(ball_mm):
    """
    Returnerer (approach_pt, exit_pt) hvis bolden er tæt på en kant,
    ellers (None, None).
    Approach-punktet er WALL_APPROACH_MM fra bolden, vinkelret på kanten.
    """
    x, y = ball_mm
    W, H = BOARD_WIDTH_MM, BOARD_HEIGHT_MM
    dists = {"left": x, "right": W - x, "bottom": y, "top": H - y}
    wall, d = min(dists.items(), key=lambda kv: kv[1])
    if d > WALL_MARGIN_MM:
        return None, None
    a = WALL_APPROACH_MM
    nx, ny = {"left": (a, 0), "right": (-a, 0), "bottom": (0, a), "top": (0, -a)}[wall]
    pt = (x + nx, y + ny)
    return pt, pt  # robotten korer ind, samler, og bakker til samme punkt


def plan_route(white_mm, orange_mm, start_mm, cross_mm):
    """
    Waypoint-liste (x_mm, y_mm).
    - Hvide bolde forst i hvert kvadrant, orange sidst.
    - Kant-bolde: vinkelret approach + exit.
    - Kryds-naere bolde: approach langs kryds-aksen.
    - Kvadrant-skift: korer til indgangspunkt GATE_OFFSET_MM fra krydset.
    """
    all_balls = [("white", p) for p in white_mm] + [("orange", p) for p in orange_mm]
    quads = {q: [] for q in [1, 2, 3, 4]}
    for kind, p in all_balls:
        quads[_quadrant(p, cross_mm)].append((kind, p))
    for q in quads:
        quads[q] = sorted(quads[q], key=lambda x: 0 if x[0] == "white" else 1)

    route = []
    current = start_mm
    cur_q = _quadrant(current, cross_mm)

    for q in [1, 2, 3, 4]:
        if not quads[q]:
            continue

        # Skift kvadrant via indgangspunkt
        if q != cur_q:
            entry = _entry_point_for_quadrant(q, cross_mm)
            for wp in _detour(current, entry, cross_mm):
                route.append(wp);
                current = wp
            cur_q = q

        pts = [p for _, p in quads[q]]
        ordered = _nearest_neighbor(current, pts)

        for nxt in ordered:
            wall_ap, wall_ex = _wall_approach(nxt)
            cross_ap, cross_ex = _approach_exit(nxt, cross_mm)

            if wall_ap is not None:
                # Kant-bold: kor hen til approach-punkt, ind til bold, bak ud
                for wp in _detour(current, wall_ap, cross_mm):
                    route.append(wp);
                    current = wp
                route.append(nxt);
                current = nxt
                route.append(wall_ex);
                current = wall_ex
            else:
                # Normal eller kryds-nær bold
                if cross_ap:
                    # Hvis bolden ligger meget tæt på krydset
                    for wp in cross_ap:
                        for ww in _detour(current, wp, cross_mm):
                            route.append(ww);
                            current = ww
                    route.append(nxt);
                    current = nxt
                    for wp in cross_ex:
                        route.append(wp);
                        current = wp
                else:
                    # FIX: Tjek ALTID for detour, også ved normale bolde der ligger frit!
                    for ww in _detour(current, nxt, cross_mm):
                        route.append(ww);
                        current = ww

    return route


def draw_route_world(world_img, route_mm, start_mm, scale=DISPLAY_SCALE):
    if not route_mm:
        return
    pts_view = [world_to_view(p, scale) for p in [start_mm] + list(route_mm)]
    for i in range(len(pts_view) - 1):
        p1, p2 = pts_view[i], pts_view[i + 1]
        cv2.line(world_img, p1, p2, (0, 255, 0), 2)
        # Pil-hoved
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        ex = int(mx - 8 * math.cos(angle - 0.4))
        ey = int(my - 8 * math.sin(angle - 0.4))
        cv2.line(world_img, (mx, my), (ex, ey), (0, 200, 0), 2)
        ex = int(mx - 8 * math.cos(angle + 0.4))
        ey = int(my - 8 * math.sin(angle + 0.4))
        cv2.line(world_img, (mx, my), (ex, ey), (0, 200, 0), 2)


def draw_direction_overlay(frame, dir_info):
    if not dir_info.get("found"):
        cv2.putText(frame, "Robot: ikke fundet", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2)
        return
    h = dir_info["heading"]
    c = dir_info["cardinal"]
    cv2.putText(frame, f"Retning: {h:+.1f} deg  ({c})", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 0), 2)
    # Tegn akse på markøren
    cv2.drawFrameAxes(frame, dir_info["cam_mat"], dir_info["dist"],
                      dir_info["rvec"], dir_info["tvec"], MARKER_SIZE_MM * 0.5)
    # Retningspil
    cx, cy = dir_info["center_px"]
    ex = int(cx + 60 * math.cos(math.radians(h)))
    ey = int(cy - 60 * math.sin(math.radians(h)))
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 0), 3, tipLength=0.3)


def draw_robot_in_world(world_img, dir_info, H_px_to_world, scale=DISPLAY_SCALE):
    if not dir_info.get("found"):
        return
    cx, cy = dir_info["center_px"]
    wmm = pixel_to_world((cx, cy), H_px_to_world)
    pv = world_to_view(wmm, scale)
    cv2.circle(world_img, pv, 10, (255, 180, 0), -1)
    h = dir_info["heading"]
    ex = int(pv[0] + 25 * math.cos(math.radians(h)))
    ey = int(pv[1] - 25 * math.sin(math.radians(h)))
    cv2.arrowedLine(world_img, pv, (ex, ey), (255, 180, 0), 2, tipLength=0.4)
    cv2.putText(world_img, f"{h:+.0f}", (pv[0] + 12, pv[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 0), 1)


def draw_balls_world(world_img, whites_px, oranges_px, H_px_to_world,
                     scale=DISPLAY_SCALE):
    for (x, y, r) in whites_px:
        wm = pixel_to_world((x, y), H_px_to_world)
        pv = world_to_view(wm, scale)
        cv2.circle(world_img, pv, 6, (0, 255, 255), -1)
        cv2.putText(world_img, f"W({int(wm[0])},{int(wm[1])})",
                    (pv[0] + 6, pv[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
    for (x, y, r) in oranges_px:
        wm = pixel_to_world((x, y), H_px_to_world)
        pv = world_to_view(wm, scale)
        cv2.circle(world_img, pv, 6, (0, 128, 255), -1)
        cv2.putText(world_img, f"O({int(wm[0])},{int(wm[1])})",
                    (pv[0] + 6, pv[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 128, 255), 1)


# ════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise SystemExit("Kan ikke åbne kamera")

cfg = DetectorConfig()
prev_corners = None

# Vedligeholdt tilstand
last_corners = None
last_H_px_world = None
last_route_mm = []
last_cross_info = None
last_dir_info = {"found": False}
last_whites_px = []
last_oranges_px = []

# Auto-update timer
ROUTE_INTERVAL_S = 3.0
last_route_time = 0.0

# Start-position i mm — opdateres fra ArUco eller kan sættes manuelt
start_mm = (BOARD_WIDTH_MM * 0.1, BOARD_HEIGHT_MM * 0.5)

for win in ["camera", "world", "ball edges"]:
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

print("Taster:  r = genberegn rute   p = print snapshot   ESC = afslut")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    H, W = frame.shape[:2]

    # ── 1. Bane-detektion ───────────────────────────────────────────────────
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)
    bp = extract_boundary_points(mask)
    mdl = fit_frame_lines(bp)

    draw_model_line(display, mdl["top"], (255, 0, 0), 3)
    draw_model_line(display, mdl["bottom"], (0, 0, 255), 3)
    draw_model_line(display, mdl["left"], (0, 255, 0), 3)
    draw_model_line(display, mdl["right"], (0, 255, 255), 3)

    world_img = np.zeros((500, 700, 3), dtype=np.uint8)
    H_px_to_world = None
    H_px_to_view = None

    if all(mdl[k] is not None for k in ["top", "bottom", "left", "right"]):
        new_corners = {
            "TL": intersect_horizontal_vertical(mdl["top"], mdl["left"]),
            "TR": intersect_horizontal_vertical(mdl["top"], mdl["right"]),
            "BL": intersect_horizontal_vertical(mdl["bottom"], mdl["left"]),
            "BR": intersect_horizontal_vertical(mdl["bottom"], mdl["right"]),
        }
        if all(v is not None for v in new_corners.values()):
            if prev_corners is None:
                corners = new_corners
            else:
                corners = {n: (
                    int(SMOOTH_ALPHA * prev_corners[n][0] + (1 - SMOOTH_ALPHA) * new_corners[n][0]),
                    int(SMOOTH_ALPHA * prev_corners[n][1] + (1 - SMOOTH_ALPHA) * new_corners[n][1]),
                ) for n in new_corners}
            prev_corners = corners
            last_corners = corners

            H_px_to_world, H_px_to_view, dw, dh = compute_homographies(
                corners, display_scale=DISPLAY_SCALE)
            last_H_px_world = H_px_to_world

            world_raw = cv2.warpPerspective(frame, H_px_to_view, (dw, dh))
            world_img = world_raw.copy()
            draw_world_grid(world_img, display_scale=DISPLAY_SCALE, step_mm=200)

            # ── 2. Kryds i world ────────────────────────────────────────────
            cross_info = detect_red_cross_world(world_raw, DISPLAY_SCALE)
            if cross_info:
                last_cross_info = cross_info
                cv2.drawContours(world_img, [cross_info["contour"]], -1, (255, 255, 255), 2)
                cv2.circle(world_img, cross_info["center_view"], 6, (0, 0, 255), -1)

    # ── 3. Bold-detektion ───────────────────────────────────────────────────
    whites_px, oranges_px, edges = detect_balls(frame, cfg)
    last_whites_px = whites_px
    last_oranges_px = oranges_px

    # Tegn bolde på camera-view
    for (x, y, r) in whites_px:
        cv2.circle(display, (x, y), r, (0, 255, 255), 2)
    for (x, y, r) in oranges_px:
        cv2.circle(display, (x, y), r, (0, 128, 255), 2)

    # Tegn bolde i world-view
    if H_px_to_world is not None:
        draw_balls_world(world_img, whites_px, oranges_px, H_px_to_world)

    # ── 4. Retnings-detektion (ArUco) ───────────────────────────────────────
    dir_info = detect_direction(frame)
    if dir_info["found"]:
        last_dir_info = dir_info
        # Brug robot-position som start hvis vi har homografi
        if H_px_to_world is not None:
            start_mm = pixel_to_world(dir_info["center_px"], H_px_to_world)

    draw_direction_overlay(display, last_dir_info)
    if H_px_to_world is not None:
        draw_robot_in_world(world_img, last_dir_info, H_px_to_world)

    # ── 5. Tegn rute i world-view ───────────────────────────────────────────
    if last_route_mm and H_px_to_world is not None:
        draw_route_world(world_img, last_route_mm, start_mm)

    # HUD
    cv2.putText(display,
                f"Bolde: {len(whites_px)}W  {len(oranges_px)}O  "
                f"| Kryds: {'OK' if last_cross_info else '?'}  "
                f"| Robot: {'OK' if last_dir_info.get('found') else '?'}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("camera", display)
    cv2.imshow("world", world_img)
    cv2.imshow("ball edges", edges)

    key = cv2.waitKey(1) & 0xFF

    # ── Auto-recalculate every ROUTE_INTERVAL_S seconds ────────────────────
    now = time.time()
    if (now - last_route_time >= ROUTE_INTERVAL_S
            and last_H_px_world is not None
            and last_cross_info is not None):
        cross_mm = last_cross_info["center_mm"]
        whites_mm = [pixel_to_world((x, y), last_H_px_world)
                     for (x, y, r) in last_whites_px]
        oranges_mm = [pixel_to_world((x, y), last_H_px_world)
                      for (x, y, r) in last_oranges_px]
        last_route_mm = plan_route(whites_mm, oranges_mm, start_mm, cross_mm)
        last_route_time = now

    # ── r: genberegn rute ───────────────────────────────────────────────────
    if key == ord("r"):
        if last_H_px_world is None:
            print("Ingen homografi endnu — vent på bane-detektion")
        elif last_cross_info is None:
            print("Kryds ikke fundet — vent på kryds-detektion")
        else:
            cross_mm = last_cross_info["center_mm"]
            whites_mm = [pixel_to_world((x, y), last_H_px_world)
                         for (x, y, r) in last_whites_px]
            oranges_mm = [pixel_to_world((x, y), last_H_px_world)
                          for (x, y, r) in last_oranges_px]
            last_route_mm = plan_route(whites_mm, oranges_mm, start_mm, cross_mm)
            print(f"Rute beregnet: {len(last_route_mm)} waypoints")
            for i, wp in enumerate(last_route_mm, 1):
                print(f"  [{i:2d}]  ({wp[0]:.1f}, {wp[1]:.1f}) mm")

    # ── p: snapshot ─────────────────────────────────────────────────────────
    elif key == ord("p"):
        print("\n════════════ SNAPSHOT ════════════")
        if last_corners and last_H_px_world:
            for name in ["TL", "TR", "BL", "BR"]:
                px = last_corners[name]
                wm = pixel_to_world(px, last_H_px_world)
                print(f"  {name}  px={px}  mm=({wm[0]:.1f}, {wm[1]:.1f})")
        if last_cross_info:
            cm = last_cross_info["center_mm"]
            print(f"  Kryds center: ({cm[0]:.1f}, {cm[1]:.1f}) mm")
        if last_dir_info.get("found"):
            print(f"  Robot retning: {last_dir_info['heading']:+.1f} deg "
                  f"({last_dir_info['cardinal']})")
            print(f"  Robot start mm: ({start_mm[0]:.1f}, {start_mm[1]:.1f})")
        print(f"  Hvide bolde:  {len(last_whites_px)}")
        print(f"  Orange bolde: {len(last_oranges_px)}")
        print(f"  Rute waypoints: {len(last_route_mm)}")
        print("══════════════════════════════════\n")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()