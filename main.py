import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_bestfit"))

import cv2
import cv2.aruco as aruco
import numpy as np
import math
import time
import threading
from dataclasses import dataclass

from red_mask import red_mask
from boundary_points import extract_boundary_points
from best_fit_lines import fit_frame_lines
from geometry import draw_model_line, intersect_horizontal_vertical
from homography import (
    compute_homographies, pixel_to_world, draw_world_grid,
    world_to_view, BOARD_WIDTH_MM, BOARD_HEIGHT_MM,
)
from robot_client import get_client

robot = get_client()
robot.connect()

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX       = 1
DISPLAY_SCALE      = 0.5
SMOOTH_ALPHA       = 0.80

ARUCO_DICT         = aruco.DICT_4X4_50
ROBOT_MARKER_ID    = 0
HEADING_OFFSET_DEG = 0.0
MARKER_SIZE_MM     = 80

WALL_MARGIN_MM     = 150
WALL_APPROACH_MM   = 180
CROSS_MARGIN_MM    = 250
POSITION_TOL_MM    = 50
APPROACH_OFFSET_MM = 80
GOAL_MIN_GAP_MM    = 60
GOAL_APPROACH_MM   = 350

TURN_TIMEOUT_S     = 0.7
ROUTE_INTERVAL_S   = 3.0
BALL_DRIVE_SPEED   = 10
REVERSE_THRESHOLD  = 181

HEADING_SMOOTH_ALPHA = 0.15
POSE_SMOOTH_ALPHA    = 0.60

# --- NEW GREEN TRACKING GEOMETRY ---
GREEN_H_MIN      = 33
GREEN_H_MAX      = 95
GREEN_S_MIN      = 45
GREEN_V_MIN      = 45
MIN_BLOB_AREA    = 5

ROBOT_WIDTH_MM   = 190.0
ROBOT_LENGTH_MM  = 288.0
HALF_W           = ROBOT_WIDTH_MM / 2.0
LENGTH           = ROBOT_LENGTH_MM
SIDE_SIGN        = 1.0
MAX_HISTORY_ASSIGN_MM = 220.0

MODEL = {
    "FL": (0.0,      -HALF_W),
    "FR": (0.0,       HALF_W),
    "BL": (-LENGTH,  -HALF_W),
    "BR": (-LENGTH,   HALF_W),
}

# --- HARDCODED GOALS ---
HARDCODED_GOALS_MM = [
    (0.0, 600.0),                             # Left wall goal
    (float(BOARD_WIDTH_MM), 600.0)            # Right wall goal
]

# Waypoint type constants
NAV  = "nav"
BALL = "ball"
GOAL = "goal"
CURRENT_ROBOT_POLYGON = None

# ════════════════════════════════════════════════════════════════════════════
# BALL DETECTION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectorConfig:
    canny_t1:     int   = 110
    canny_t2:     int   = 300
    blur_k:       int   = 31
    blur_sigma:   int   = 1
    dilate_k:     int   = 3
    dilate_it:    int   = 2
    dp:           float = 1.2
    min_dist:     int   = 18
    param2:       int   = 18
    min_radius:   int   = 6
    max_radius:   int   = 11
    white_s_max:  int   = 130
    orange_h_min: int   = 12
    orange_h_max: int   = 35
    orange_s_min: int   = 70
    orange_v_min: int   = 85


def _ensure_odd(n):
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def detect_balls(frame, cfg):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (_ensure_odd(cfg.blur_k),) * 2, cfg.blur_sigma)
    edges = cv2.Canny(gray, cfg.canny_t1, cfg.canny_t2)
    if cfg.dilate_it > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (_ensure_odd(cfg.dilate_k),) * 2)
        edges = cv2.dilate(edges, k, iterations=cfg.dilate_it)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H_ch, S, V = cv2.split(hsv)
    white_mask  = ((S < cfg.white_s_max) & (V > 120)).astype(np.uint8) * 255
    orange_mask = ((H_ch >= cfg.orange_h_min) & (H_ch <= cfg.orange_h_max) &
                   (S  >= cfg.orange_s_min)   & (V    >= cfg.orange_v_min)
                   ).astype(np.uint8) * 255

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=cfg.dp, minDist=cfg.min_dist,
        param1=cfg.canny_t2, param2=cfg.param2,
        minRadius=cfg.min_radius, maxRadius=cfg.max_radius,
    )

    whites, oranges = [], []
    if circles is not None:
        h_f, w_f = frame.shape[:2]
        seen = []
        for (cx, cy, r) in np.round(circles[0]).astype(int):
            if any((cx-sx)**2 + (cy-sy)**2 < 9 and abs(r-sr) <= 1
                   for sx, sy, sr in seen):
                continue
            seen.append((cx, cy, r))
            x0, x1 = max(0, cx-r), min(w_f, cx+r)
            y0, y1 = max(0, cy-r), min(h_f, cy+r)
            if x1 <= x0 or y1 <= y0:
                continue
            wo = np.mean(white_mask [y0:y1, x0:x1] > 0)
            oo = np.mean(orange_mask[y0:y1, x0:x1] > 0)
            (oranges if oo > wo else whites).append((cx, cy, r))

    return whites, oranges, edges


# ════════════════════════════════════════════════════════════════════════════
# CROSS & GOAL DETECTION
# ════════════════════════════════════════════════════════════════════════════

def detect_cross(world_img_raw, scale=DISPLAY_SCALE):
    hsv  = cv2.cvtColor(world_img_raw, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)
    h_i, w_i = mask.shape

    roi = np.zeros_like(mask)
    y0, y1 = int(0.25*h_i), int(0.75*h_i)
    x0, x1 = int(0.25*w_i), int(0.75*w_i)
    roi[y0:y1, x0:x1] = mask[y0:y1, x0:x1]

    k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN,  k)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return None

    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-8:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return {
        "center_mm":   (cx / scale, BOARD_HEIGHT_MM - cy / scale),
        "center_view": (int(cx), int(cy)),
        "contour":     cnt,
    }

def _find_gaps(has_red, min_len):
    gaps, start = [], None
    for i, v in enumerate(has_red):
        if not v and start is None:
            start = i
        elif v and start is not None:
            if i - start >= min_len:
                gaps.append((start, i))
            start = None
    if start is not None and len(has_red) - start >= min_len:
        gaps.append((start, len(has_red)))
    return gaps

def detect_goals(world_img_raw, scale=DISPLAY_SCALE):
    hsv  = cv2.cvtColor(world_img_raw, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)
    h_i, w_i = mask.shape
    border  = 20
    min_gap = max(3, int(GOAL_MIN_GAP_MM * scale))
    goals   = []

    strip = mask[max(0, h_i - border):h_i, :]
    for gx0, gx1 in _find_gaps(np.any(strip > 0, axis=0), min_gap):
        goals.append(((gx0 + gx1) / 2.0 / scale, 0.0))

    strip = mask[0:border, :]
    for gx0, gx1 in _find_gaps(np.any(strip > 0, axis=0), min_gap):
        goals.append(((gx0 + gx1) / 2.0 / scale, float(BOARD_HEIGHT_MM)))

    strip = mask[:, 0:border]
    for gy0, gy1 in _find_gaps(np.any(strip > 0, axis=1), min_gap):
        goals.append((0.0, BOARD_HEIGHT_MM - (gy0 + gy1) / 2.0 / scale))

    strip = mask[:, max(0, w_i - border):w_i]
    for gy0, gy1 in _find_gaps(np.any(strip > 0, axis=1), min_gap):
        goals.append((float(BOARD_WIDTH_MM),
                      BOARD_HEIGHT_MM - (gy0 + gy1) / 2.0 / scale))

    return goals


# ════════════════════════════════════════════════════════════════════════════
# ARUCO HEADING
# ════════════════════════════════════════════════════════════════════════════

_aruco_dict     = aruco.getPredefinedDictionary(ARUCO_DICT)
_aruco_params   = aruco.DetectorParameters()
_aruco_detector = aruco.ArucoDetector(_aruco_dict, _aruco_params)
_half           = MARKER_SIZE_MM / 2.0
_OBJ_PTS        = np.array([
    [-_half,  _half, 0],
    [ _half,  _half, 0],
    [ _half, -_half, 0],
    [-_half, -_half, 0],
], dtype=np.float32)

_BOARD_OBJ_PTS = np.array([
    [0,              0,               0],
    [0,              BOARD_HEIGHT_MM, 0],
    [BOARD_WIDTH_MM, BOARD_HEIGHT_MM, 0],
    [BOARD_WIDTH_MM, 0,               0],
], dtype=np.float32)

def _line_intersect(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9: return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / d
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / d
    return px, py

def estimate_focal_length(corners_px, principal_point):
    TL, TR = corners_px["TL"], corners_px["TR"]
    BL, BR = corners_px["BL"], corners_px["BR"]
    vp1 = _line_intersect(TL, TR, BL, BR)
    vp2 = _line_intersect(TL, BL, TR, BR)
    if vp1 is None or vp2 is None: return None
    px, py = principal_point
    dot = (vp1[0]-px)*(vp2[0]-px) + (vp1[1]-py)*(vp2[1]-py)
    f_sq = -dot
    if f_sq <= 0: return None
    f = math.sqrt(f_sq)
    diag = math.hypot(2*px, 2*py)
    if not (0.3*diag <= f <= 6*diag): return None
    return f

_focal_length_est = None

def _camera_matrix(frame_shape):
    H, W = frame_shape[:2]
    f    = _focal_length_est if _focal_length_est is not None else max(W, H)
    cam  = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    return cam, dist

def board_pose(corners_px, cam_mat, dist_coef):
    img_pts = np.array([
        corners_px["BL"], corners_px["TL"], corners_px["TR"], corners_px["BR"],
    ], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        _BOARD_OBJ_PTS, img_pts, cam_mat, dist_coef,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok: return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec

def marker_ground_position(tvec_marker, pose):
    R, t = pose
    p_world = (R.T @ (tvec_marker - t)).flatten()
    return float(p_world[0]), float(p_world[1])

def detect_heading(frame):
    cam, dist = _camera_matrix(frame.shape)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _aruco_detector.detectMarkers(gray)
    if ids is None: return {"found": False}
    for i, mid in enumerate(ids.flatten()):
        if mid != ROBOT_MARKER_ID: continue
        mc = corners[i][0]
        ok, rvec, tvec = cv2.solvePnP(
            _OBJ_PTS, mc.astype(np.float32), cam, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok: continue
        R, _ = cv2.Rodrigues(rvec)
        heading = math.degrees(math.atan2(-R[1, 0], R[0, 0])) + HEADING_OFFSET_DEG
        return {
            "found":     True,
            "heading":   heading,
            "center_px": (int(mc[:, 0].mean()), int(mc[:, 1].mean())),
            "rvec":      rvec,
            "tvec":      tvec,
            "cam_mat":   cam,
            "dist_coef": dist,
        }
    return {"found": False}


# ════════════════════════════════════════════════════════════════════════════
# NEW RIGID BODY GREEN BLOB TRACKING
# ════════════════════════════════════════════════════════════════════════════

def add2(a, b): return (a[0] + b[0], a[1] + b[1])
def sub2(a, b): return (a[0] - b[0], a[1] - b[1])
def mul2(v, k): return (v[0] * k, v[1] * k)
def dot2(a, b): return a[0] * b[0] + a[1] * b[1]
def dist2(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
def norm2(v):
    n = math.hypot(v[0], v[1])
    return (0.0, 0.0) if n < 1e-9 else (v[0] / n, v[1] / n)
def midpoint(a, b): return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
def heading_unit(heading_deg):
    a = math.radians(heading_deg)
    return (math.cos(a), math.sin(a))
def side_unit_from_heading(heading_deg):
    f = heading_unit(heading_deg)
    return (SIDE_SIGN * (-f[1]), SIDE_SIGN * f[0])

def world_from_top(top_center, label, heading_deg):
    f = heading_unit(heading_deg)
    s = side_unit_from_heading(heading_deg)
    xf, ys = MODEL[label]
    return (top_center[0] + xf * f[0] + ys * s[0], top_center[1] + xf * f[1] + ys * s[1])

def top_from_known_marker(pt_world, label, heading_deg):
    f = heading_unit(heading_deg)
    s = side_unit_from_heading(heading_deg)
    xf, ys = MODEL[label]
    return (pt_world[0] - xf * f[0] - ys * s[0], pt_world[1] - xf * f[1] - ys * s[1])

def solve_from_any_labeled_points(labels_found, heading_deg):
    usable = {k: v for k, v in labels_found.items() if k in MODEL}
    if not usable: return None, None
    top_candidates = [top_from_known_marker(pt, label, heading_deg) for label, pt in usable.items()]
    tx = sum(p[0] for p in top_candidates) / len(top_candidates)
    ty = sum(p[1] for p in top_candidates) / len(top_candidates)
    top_center = (tx, ty)
    solved = {}
    for label in MODEL.keys():
        solved[label] = usable[label] if label in usable else world_from_top(top_center, label, heading_deg)
    return solved, top_center

def compute_front_normal(fl, fr, heading_deg):
    edge = (fr[0] - fl[0], fr[1] - fl[1])
    n1 = norm2((-edge[1], edge[0]))
    n2 = (-n1[0], -n1[1])
    h = heading_unit(heading_deg)
    return n2 if dot2(n2, h) > dot2(n1, h) else n1

def detect_green_blobs(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (GREEN_H_MIN, GREEN_S_MIN, GREEN_V_MIN), (GREEN_H_MAX, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_BLOB_AREA: continue
        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-6: continue
        blobs.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
    return blobs

def classify_visible_blobs(blobs_world, heading_deg, aruco_center_world):
    f = heading_unit(heading_deg)
    s = side_unit_from_heading(heading_deg)
    pts = [{"pt": p, "pf": dot2(p, f), "ps": dot2(p, s)} for p in blobs_world]
    n = len(pts)
    labels = {}
    if n == 0: return {}, "none"

    if n == 4:
        pts_sorted = sorted(pts, key=lambda t: t["pf"], reverse=True)
        front = sorted(pts_sorted[:2], key=lambda t: t["ps"])
        back = sorted(pts_sorted[2:], key=lambda t: t["ps"])
        labels["FL"] = front[0]["pt"]; labels["FR"] = front[1]["pt"]
        labels["BL"] = back[0]["pt"];  labels["BR"] = back[1]["pt"]
        return labels, "4pts"

    if n == 3:
        pts_sorted = sorted(pts, key=lambda t: t["pf"], reverse=True)
        span_front = abs(pts_sorted[0]["pf"] - pts_sorted[1]["pf"])
        span_back = abs(pts_sorted[1]["pf"] - pts_sorted[2]["pf"])
        if span_front < span_back:
            front = sorted(pts_sorted[:2], key=lambda t: t["ps"])
            labels["FL"] = front[0]["pt"]; labels["FR"] = front[1]["pt"]
            rel = sub2(pts_sorted[2]["pt"], aruco_center_world)
            labels["BL" if dot2(rel, s) < 0 else "BR"] = pts_sorted[2]["pt"]
            return labels, "3pts_frontpair"
        else:
            back = sorted(pts_sorted[1:], key=lambda t: t["ps"])
            labels["BL"] = back[0]["pt"]; labels["BR"] = back[1]["pt"]
            rel = sub2(pts_sorted[0]["pt"], aruco_center_world)
            labels["FL" if dot2(rel, s) < 0 else "FR"] = pts_sorted[0]["pt"]
            return labels, "3pts_backpair"

    if n == 2:
        d = dist2(pts[0]["pt"], pts[1]["pt"])
        diag = math.hypot(ROBOT_WIDTH_MM, ROBOT_LENGTH_MM)
        if abs(d - ROBOT_WIDTH_MM) < 45:
            pair = sorted(pts, key=lambda t: t["ps"])
            rel = sub2(midpoint(pair[0]["pt"], pair[1]["pt"]), aruco_center_world)
            if dot2(rel, f) >= 0:
                labels["FL"] = pair[0]["pt"]; labels["FR"] = pair[1]["pt"]
                return labels, "2pts_frontrow"
            else:
                labels["BL"] = pair[0]["pt"]; labels["BR"] = pair[1]["pt"]
                return labels, "2pts_backrow"
        if abs(d - ROBOT_LENGTH_MM) < 50:
            pair = sorted(pts, key=lambda t: t["pf"], reverse=True)
            rel = sub2(midpoint(pair[0]["pt"], pair[1]["pt"]), aruco_center_world)
            if dot2(rel, s) < 0:
                labels["FL"] = pair[0]["pt"]; labels["BL"] = pair[1]["pt"]
            else:
                labels["FR"] = pair[0]["pt"]; labels["BR"] = pair[1]["pt"]
            return labels, "2pts_side"
        if abs(d - diag) < 60:
            pair = sorted(pts, key=lambda t: t["pf"], reverse=True)
            rel = sub2(pair[0]["pt"], aruco_center_world)
            if dot2(rel, s) < 0:
                labels["FL"] = pair[0]["pt"]; labels["BR"] = pair[1]["pt"]
            else:
                labels["FR"] = pair[0]["pt"]; labels["BL"] = pair[1]["pt"]
            return labels, "2pts_diag"
        return {}, "2pts_unknown"

    if n == 1:
        p = pts[0]["pt"]
        rel = sub2(p, aruco_center_world)
        is_front = dot2(rel, f) >= 0
        is_left = dot2(rel, s) < 0
        if is_front and is_left: labels["FL"] = p
        elif is_front and not is_left: labels["FR"] = p
        elif not is_front and is_left: labels["BL"] = p
        else: labels["BR"] = p
        return labels, "1pt"

    return {}, "unknown"

def assign_using_last_pose(blobs_world, last_solved_markers, max_dist=MAX_HISTORY_ASSIGN_MM):
    if last_solved_markers is None: return {}
    labels_found, used_labels = {}, set()
    for p in blobs_world:
        best_lab, best_d = None, float("inf")
        for lab, prev_pt in last_solved_markers.items():
            if lab in used_labels: continue
            d = dist2(p, prev_pt)
            if d < best_d: best_d = d; best_lab = lab
        if best_lab is not None and best_d <= max_dist:
            labels_found[best_lab] = p
            used_labels.add(best_lab)
    return labels_found

def solve_robot_from_labels(labels_found, case_name, heading_deg, last_top_center=None, last_solved_markers=None):
    if case_name in ("4pts", "3pts_frontpair", "3pts_backpair", "2pts_frontrow", "2pts_backrow", "2pts_side", "2pts_diag", "1pt"):
        return solve_from_any_labeled_points(labels_found, heading_deg)
    if any(k in MODEL for k in labels_found.keys()):
        return solve_from_any_labeled_points(labels_found, heading_deg)
    return None, None


# ════════════════════════════════════════════════════════════════════════════
# ROUTE PLANNING
# ════════════════════════════════════════════════════════════════════════════

def _dist(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

def point_inside_robot(point, solved_markers):
    """
    Returns True if a world point lies inside the robot rectangle.

    solved_markers must contain:
        FL FR BL BR
    """

    if solved_markers is None:
        return False

    poly = np.array([
        solved_markers["FL"],
        solved_markers["FR"],
        solved_markers["BR"],
        solved_markers["BL"],
    ], dtype=np.float32)

    return cv2.pointPolygonTest(
        poly,
        (float(point[0]), float(point[1])),
        False
    ) >= 0
    
def _seg_dist(p1, p2, pt):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if dx == dy == 0: return _dist(p1, pt)
    t = max(0.0, min(1.0, ((pt[0]-p1[0])*dx + (pt[1]-p1[1])*dy) / (dx*dx + dy*dy)))
    return _dist((p1[0] + t*dx, p1[1] + t*dy), pt)

def _detour(p1, p2, cross_mm):
    if _seg_dist(p1, p2, cross_mm) >= CROSS_MARGIN_MM: return [p2]
    cx, cy = cross_mm
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    lsq = dx*dx + dy*dy
    if lsq < 1e-9: return [p2]
    t = max(0.0, min(1.0, ((cx - p1[0])*dx + (cy - p1[1])*dy) / lsq))
    clx, cly = p1[0] + t*dx, p1[1] + t*dy
    vx, vy = clx - cx, cly - cy
    mag = math.hypot(vx, vy)
    if mag < 1e-9:
        pl = math.hypot(dx, dy)
        vx, vy, mag = -dy / pl, dx / pl, 1.0
    push   = CROSS_MARGIN_MM * 1.8
    return [(cx + vx / mag * push, cy + vy / mag * push), p2]

def _wall_approach(ball_mm):
    x, y   = ball_mm
    apd    = WALL_APPROACH_MM
    margin = WALL_MARGIN_MM
    near_L = x < margin; near_R = BOARD_WIDTH_MM - x < margin
    near_B = y < margin; near_T = BOARD_HEIGHT_MM - y < margin

    if near_L and near_B: return (x + apd, y + apd)
    if near_R and near_B: return (x - apd, y + apd)
    if near_L and near_T: return (x + apd, y - apd)
    if near_R and near_T: return (x - apd, y - apd)
    if near_L: return (x + apd, y)
    if near_R: return (x - apd, y)
    if near_B: return (x, y + apd)
    if near_T: return (x, y - apd)
    return None

def _goal_approach(goal_mm):
    x, y = goal_mm
    d = GOAL_APPROACH_MM
    if x < 10: return (d, y)
    if x > BOARD_WIDTH_MM - 10: return (BOARD_WIDTH_MM - d, y)
    if y < 10: return (x, d)
    if y > BOARD_HEIGHT_MM - 10: return (x, BOARD_HEIGHT_MM - d)
    return None

def _add_waypoints(route, current, target, cross_mm, wp_type):
    path = _detour(current, target, cross_mm)
    for wp in path[:-1]: route.append((wp[0], wp[1], NAV))
    route.append((target[0], target[1], wp_type))
    return target

def plan_route(white_mm, orange_mm, robot_mm, cross_mm, goals_mm):
    route   = []
    current = robot_mm
    
    # Filter: Keep only free balls
    free_white = [b for b in white_mm if _wall_approach(b) is None]
    free_orange = [b for b in orange_mm if _wall_approach(b) is None]

    def add_ball(ball):
        nonlocal current
        current = _add_waypoints(route, current, ball, cross_mm, BALL)

    remaining = list(free_white)
    while remaining:
        nxt = min(remaining, key=lambda p: _dist(current, p))
        remaining.remove(nxt)
        add_ball(nxt)

    for nxt in free_orange: add_ball(nxt)

    if not free_white and not free_orange and goals_mm:
        goal     = min(goals_mm, key=lambda g: _dist(current, g))
        approach = _goal_approach(goal)
        
        # Only add the approach waypoint if the robot is far away from it
        if approach and _dist(current, approach) > 100: 
            current = _add_waypoints(route, current, approach, cross_mm, NAV)
            
        _add_waypoints(route, current, goal, cross_mm, GOAL)

    return route


# ════════════════════════════════════════════════════════════════════════════
# DRAWING
# ════════════════════════════════════════════════════════════════════════════

def draw_route(world_img, route, start_mm, scale=DISPLAY_SCALE):
    if not route: return
    color_map = {NAV: (80, 180, 80), BALL: (0, 255, 255), GOAL: (0, 140, 255)}
    pts = [world_to_view(start_mm, scale)] + [world_to_view((p[0], p[1]), scale) for p in route]
    for i in range(len(pts) - 1):
        col = color_map.get(route[i][2], (0, 255, 0))
        cv2.line(world_img, pts[i], pts[i+1], col, 2)

def draw_robot_solution(world_img, solved_markers, top_center, heading_deg, display_scale, case_name, measured_labels=None):
    if measured_labels is None: measured_labels = set(solved_markers.keys())
    color_measured  = {"FL": (0, 255, 255), "FR": (255, 255, 0), "BL": (255, 0, 255), "BR": (0, 165, 255)}
    color_estimated = {"FL": (0, 180, 180), "FR": (180, 180, 0), "BL": (180, 0, 180), "BR": (0, 110, 180)}

    for label in ["FL", "FR", "BL", "BR"]:
        if label not in solved_markers: continue
        p_view = world_to_view(solved_markers[label], display_scale)
        measured = label in measured_labels
        col = color_measured[label] if measured else color_estimated[label]
        cv2.circle(world_img, p_view, 7 if measured else 5, col, -1 if measured else 2)
        cv2.putText(world_img, f"{label}{'' if measured else '*'}", (p_view[0]+8, p_view[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    fl, fr = solved_markers.get("FL"), solved_markers.get("FR")
    if fl and fr:
        cv2.line(world_img, world_to_view(fl, display_scale), world_to_view(fr, display_scale), (0, 255, 0), 2)
        forward_n = compute_front_normal(fl, fr, heading_deg)
        ex = top_center[0] + 120.0 * forward_n[0]
        ey = top_center[1] + 120.0 * forward_n[1]
        cv2.arrowedLine(world_img, world_to_view(top_center, display_scale), world_to_view((ex, ey), display_scale), (0, 0, 255), 3)

    cv2.circle(world_img, world_to_view(top_center, display_scale), 8, (255, 0, 0), -1)

def draw_balls(world_img, whites_px, oranges_px, H_px_to_world, scale=DISPLAY_SCALE):
    for (x, y, r) in whites_px:
        pv = world_to_view(pixel_to_world((x, y), H_px_to_world), scale)
        cv2.circle(world_img, pv, 6, (0, 255, 255), -1)
    for (x, y, r) in oranges_px:
        pv = world_to_view(pixel_to_world((x, y), H_px_to_world), scale)
        cv2.circle(world_img, pv, 6, (0, 128, 255), -1)

def draw_goals(world_img, goals_mm, scale=DISPLAY_SCALE):
    for (x, y) in goals_mm:
        pv = world_to_view((x, y), scale)
        cv2.circle(world_img, pv, 10, (0, 200, 100), 3)
        cv2.putText(world_img, "GOAL", (pv[0]+5, pv[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 100), 1)


# ════════════════════════════════════════════════════════════════════════════
# CAMERA + SHARED STATE
# ════════════════════════════════════════════════════════════════════════════

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise SystemExit("Cannot open camera {}".format(CAMERA_INDEX))

cfg          = DetectorConfig()
prev_corners = None

last_H_px_world = None
last_board_pose = None
last_cross      = None
last_goals      = HARDCODED_GOALS_MM
last_dir_info   = {"found": False}
last_whites_px  = []
last_oranges_px = []
last_route      = []
start_mm        = (BOARD_WIDTH_MM * 0.1, BOARD_HEIGHT_MM * 0.5)
last_route_time = 0.0
last_pos_source = "none"

# --- NEW RIGID BODY STATE ---
LAST_TOP_CENTER = None
LAST_SOLVED_MARKERS = None


# ════════════════════════════════════════════════════════════════════════════
# ROBOT EXECUTOR THREAD
# ════════════════════════════════════════════════════════════════════════════

robot_running  = threading.Event()
robot_stop_req = threading.Event()
current_route  = []
route_lock     = threading.Lock()

def _replan_from_camera():
    global last_route
    wait_end = time.time() + 3.0
    while last_H_px_world is None and time.time() < wait_end: time.sleep(0.1)

    if last_H_px_world is None:
        print("[robot] No homography - cannot replan, stopping")
        robot_running.clear(); return

    cross_mm = (last_cross["center_mm"] if last_cross else (BOARD_WIDTH_MM / 2.0, BOARD_HEIGHT_MM / 2.0))
    w_mm = [pixel_to_world((x, y), last_H_px_world) for (x, y, r) in last_whites_px]
    o_mm = [pixel_to_world((x, y), last_H_px_world) for (x, y, r) in last_oranges_px]
    new_route = plan_route(w_mm, o_mm, start_mm, cross_mm, last_goals)
    last_route = new_route
    with route_lock:
        current_route.clear()
        current_route.extend(new_route)

    if new_route: print(f"[robot] Replanned: {len(new_route)} waypoints")
    else:
        print("[robot] All balls collected - heading to goal")
        robot.stop(); robot_running.clear()

def robot_executor():
    while True:
        robot_running.wait()
        with route_lock:
            if not current_route:
                robot_running.clear(); continue
            wp = current_route.pop(0)

        if robot_stop_req.is_set(): robot_running.clear(); robot_stop_req.clear(); continue

        pos, wp_pos, wp_type = start_mm, (wp[0], wp[1]), wp[2]

        if wp_type != NAV and last_cross is not None:
            detour_pts = _detour(pos, wp_pos, last_cross["center_mm"])
            if len(detour_pts) > 1:
                bypass_pt = detour_pts[0]
                with route_lock: current_route.insert(0, wp)
                wp, wp_pos, wp_type = (bypass_pt[0], bypass_pt[1], NAV), bypass_pt, NAV

        dx, dy = wp_pos[0] - pos[0], wp_pos[1] - pos[1]
        if math.hypot(dx, dy) >= POSITION_TOL_MM:
            target_h, drive_sign = math.degrees(math.atan2(dy, dx)), 1
            current_h = last_dir_info.get("heading")

            if current_h is not None:
                # Check for reverse if NAV
                if wp_type == NAV:
                    if abs((target_h - current_h + 180) % 360 - 180) > REVERSE_THRESHOLD:
                        target_h, drive_sign = (target_h + 180) % 360, -1

                # Calculate the shortest angle difference for turn pulse
                angle_diff = abs((target_h - current_h + 180) % 360 - 180)
                angle_diff = abs((target_h - current_h + 180) % 360 - 180)
                time_wheel_spinning = 3000
                if angle_diff < 30:
                    time_wheel_spinning = 1700
                elif angle_diff < 10:
                    time_wheel_spinning = 400

            if not robot.turn_to_heading(target_h, lambda: last_dir_info.get("heading"), pulse_ms=time_wheel_spinning, timeout=TURN_TIMEOUT_S, stop_fn=lambda: robot_stop_req.is_set()):
                if robot_stop_req.is_set(): robot_running.clear(); robot_stop_req.clear(); continue
                with route_lock: current_route.insert(0, wp)
                time.sleep(0.5); continue

            # --- DYNAMIC TOLERANCE ---
            if wp_type == GOAL:
                tol = 130  # High tolerance so it doesn't have to perfectly hit the physical wall
            elif wp_type == BALL:
                tol = APPROACH_OFFSET_MM
            else:
                tol = POSITION_TOL_MM

            speed = 20
            if wp_type == BALL: speed = 40
            
            robot.drive_to_position(wp_pos, lambda: start_mm, reverse=(drive_sign < 0), tol_mm=tol, speed=speed, timeout=6.0, stop_fn=lambda: robot_stop_req.is_set(), get_heading_fn=lambda: last_dir_info.get("heading"))

        if robot_stop_req.is_set(): robot_running.clear(); robot_stop_req.clear(); continue

        if wp_type == NAV:
            robot.motor_stop()
            time.sleep(0.25)
            
            # --- FIX: Prevent infinite replan loops when heading to the goal ---
            with route_lock:
                next_wp = current_route[0] if current_route else None
            # Only recalculate if the next target is a ball
            if next_wp and next_wp[2] == BALL:
                _replan_from_camera()
            continue
            
        elif wp_type == BALL: 
            time.sleep(1.5)
            _replan_from_camera()
            
        elif wp_type == GOAL: 
            robot.eject()
            robot_running.clear()

threading.Thread(target=robot_executor, daemon=True).start()


# ════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════

for win in ["camera", "world", "edges"]: cv2.namedWindow(win, cv2.WINDOW_NORMAL)
print("Keys:  r=route  g=go  s=stop  c=collect  e=eject  p=status  ESC=quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    display = frame.copy()

    # ── 1. Board detection ──────────────────────────────────────────────────
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bp, mdl = extract_boundary_points(red_mask(hsv)), fit_frame_lines(extract_boundary_points(red_mask(hsv)))

    for k, col in zip(["top", "bottom", "left", "right"], [(255,0,0), (0,0,255), (0,255,0), (0,255,255)]):
        draw_model_line(display, mdl[k], col, 2)

    disp_h, disp_w = int(BOARD_HEIGHT_MM * DISPLAY_SCALE), int(BOARD_WIDTH_MM * DISPLAY_SCALE)
    world_img = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)

    if all(mdl[k] is not None for k in ["top", "bottom", "left", "right"]):
        new_corners = {
            "TL": intersect_horizontal_vertical(mdl["top"], mdl["left"]),
            "TR": intersect_horizontal_vertical(mdl["top"], mdl["right"]),
            "BL": intersect_horizontal_vertical(mdl["bottom"], mdl["left"]),
            "BR": intersect_horizontal_vertical(mdl["bottom"], mdl["right"]),
        }
        if all(v is not None for v in new_corners.values()):
            if prev_corners is None: corners = new_corners
            else: corners = {n: (int(SMOOTH_ALPHA * prev_corners[n][0] + (1 - SMOOTH_ALPHA) * new_corners[n][0]), int(SMOOTH_ALPHA * prev_corners[n][1] + (1 - SMOOTH_ALPHA) * new_corners[n][1])) for n in new_corners}
            prev_corners = corners

            last_H_px_world, H_px_to_view, dw, dh = compute_homographies(corners, DISPLAY_SCALE)
            f_est = estimate_focal_length(corners, (frame.shape[1] / 2.0, frame.shape[0] / 2.0))
            if f_est is not None: _focal_length_est = f_est if _focal_length_est is None else SMOOTH_ALPHA * _focal_length_est + (1 - SMOOTH_ALPHA) * f_est

            pose = board_pose(corners, *_camera_matrix(frame.shape))
            if pose is not None: last_board_pose = pose

            world_raw = cv2.warpPerspective(frame, H_px_to_view, (dw, dh))
            world_img = world_raw.copy()
            draw_world_grid(world_img, DISPLAY_SCALE, step_mm=200)

            cross = detect_cross(world_raw, DISPLAY_SCALE)
            if cross: last_cross = cross
            if last_cross: cv2.circle(world_img, last_cross["center_view"], 6, (0, 0, 255), -1)

            draw_goals(world_img, last_goals)

    # ── 4. Green blobs ───────────────────────────────────────────────────────
    green_blobs_px = detect_green_blobs(frame)
    for p in green_blobs_px: cv2.circle(display, p, 8, (0, 255, 0), 2)

    # ── 5. Balls ─────────────────────────────────────────────────────────────
    whites_px, oranges_px, edges = detect_balls(frame, cfg)
    if green_blobs_px:
        whites_px = [(cx, cy, r) for cx, cy, r in whites_px if not any(math.hypot(cx - gx, cy - gy) < r + 20 for gx, gy in green_blobs_px)]
    last_whites_px, last_oranges_px = whites_px, oranges_px

    for (x, y, r) in whites_px: cv2.circle(display, (x, y), r, (0, 255, 255), 2)
    for (x, y, r) in oranges_px: cv2.circle(display, (x, y), r, (0, 128, 255), 2)
    if last_H_px_world is not None: draw_balls(world_img, whites_px, oranges_px, last_H_px_world)

    # ── 6. Heading + POSITION TRACKING ─────────────────────────────────────────
    dir_info = detect_heading(frame)
    if dir_info["found"]:
        if last_dir_info.get("found"):
            prev_h = last_dir_info["heading"]
            diff   = (dir_info["heading"] - prev_h + 180) % 360 - 180
            dir_info["heading"] = prev_h + (1 - HEADING_SMOOTH_ALPHA) * diff
        last_dir_info = dir_info

        aruco_center_world = None
        if last_H_px_world is not None:
            acw = pixel_to_world(dir_info["center_px"], last_H_px_world)
            if math.isfinite(acw[0]) and math.isfinite(acw[1]):
                aruco_center_world = acw
        new_pos = None

        if last_H_px_world is not None and aruco_center_world is not None:
            blobs_world = [pixel_to_world(p, last_H_px_world) for p in green_blobs_px]
            labels_found, case_name = classify_visible_blobs(blobs_world, dir_info["heading"], aruco_center_world)
            measured_labels = set(k for k in labels_found.keys() if k in MODEL)

            solved_markers, top_center = solve_robot_from_labels(labels_found, case_name, dir_info["heading"], LAST_TOP_CENTER, LAST_SOLVED_MARKERS)

            # History Fallback Trigger
            if (solved_markers is None or top_center is None) and LAST_SOLVED_MARKERS is not None and len(blobs_world) > 0:
                hist_labels = assign_using_last_pose(blobs_world, LAST_SOLVED_MARKERS)
                if hist_labels:
                    solved_markers, top_center = solve_from_any_labeled_points(hist_labels, dir_info["heading"])
                    measured_labels = set(hist_labels.keys())
                    case_name = "history"

            if solved_markers is not None and top_center is not None:
                # Safely check that the math didn't produce NaN before accepting it
                if math.isfinite(top_center[0]) and math.isfinite(top_center[1]):
                    LAST_TOP_CENTER = top_center
                    LAST_SOLVED_MARKERS = dict(solved_markers)
                    CURRENT_ROBOT_POLYGON = dict(solved_markers)
                    new_pos = top_center  # Setting position to the FRONT of the robot!
                    last_pos_source = f"green({case_name})"
                    draw_robot_solution(world_img, solved_markers, top_center, dir_info["heading"], DISPLAY_SCALE, case_name, measured_labels)
            elif last_board_pose is not None:
                new_pos = marker_ground_position(dir_info["tvec"], last_board_pose)
                last_pos_source = "aruco"
            else:
                new_pos = aruco_center_world
                last_pos_source = "raw"
        elif last_board_pose is not None:
            new_pos = marker_ground_position(dir_info["tvec"], last_board_pose)
            last_pos_source = "aruco"

        # Position EMA smoothing applied to the new Top Center
        if new_pos is not None and math.isfinite(new_pos[0]) and math.isfinite(new_pos[1]):
            if math.isfinite(start_mm[0]) and math.isfinite(start_mm[1]):
                start_mm = (
                    POSE_SMOOTH_ALPHA * start_mm[0] + (1 - POSE_SMOOTH_ALPHA) * new_pos[0],
                    POSE_SMOOTH_ALPHA * start_mm[1] + (1 - POSE_SMOOTH_ALPHA) * new_pos[1],
                )
            else: start_mm = new_pos

    cv2.putText(display, f"Pos:({start_mm[0]:.0f},{start_mm[1]:.0f})mm  src:{last_pos_source}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1)

    # ── 7. Route display ───────────────────────────────────────────────────────
    now = time.time()
    if now - last_route_time >= ROUTE_INTERVAL_S and last_H_px_world is not None and last_cross is not None and not robot_running.is_set():
        w_mm = [pixel_to_world((x, y), last_H_px_world) for (x, y, r) in whites_px]
        o_mm = [pixel_to_world((x, y), last_H_px_world) for (x, y, r) in oranges_px]
        last_route = plan_route(w_mm, o_mm, start_mm, last_cross["center_mm"], last_goals)
        last_route_time = now

    draw_route(world_img, last_route, start_mm)

    cv2.imshow("camera", display)
    cv2.imshow("world",  world_img)
    cv2.imshow("edges",  edges)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        if robot_running.is_set(): print("Robot running - press s first")
        elif last_H_px_world is None or last_cross is None: print("Waiting for board + cross detection")
        else:
            w_mm = [pixel_to_world((x, y), last_H_px_world) for (x, y, r) in whites_px]
            o_mm = [pixel_to_world((x, y), last_H_px_world) for (x, y, r) in oranges_px]
            last_route = plan_route(w_mm, o_mm, start_mm, last_cross["center_mm"], last_goals)
            with route_lock: current_route.clear(); current_route.extend(last_route)
            print(f"Route: {len(last_route)} waypoints - press g to start")
    elif key == ord("g"):
        if not robot.connected: print("Robot not connected")
        elif not current_route: print("No route - press r first")
        else: robot_stop_req.clear(); robot_running.set(); print("GO")
    elif key == ord("s"): robot_stop_req.set(); robot_running.clear(); robot.stop(); print("STOP")
    elif key == ord("c"): robot.collect(); print("Collect")
    elif key == ord("e"): robot.eject(); print("Eject")
    elif key == 27: break

cap.release()
cv2.destroyAllWindows()