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

CAMERA_INDEX       = 0
DISPLAY_SCALE      = 0.5
SMOOTH_ALPHA       = 0.80

ARUCO_DICT         = aruco.DICT_4X4_50
ROBOT_MARKER_ID    = 0
HEADING_OFFSET_DEG = -90
MARKER_SIZE_MM     = 80

WALL_MARGIN_MM     = 200    # ball within this distance of a wall = wall ball
WALL_APPROACH_MM   = 180    # perpendicular approach point distance from wall ball
CROSS_MARGIN_MM    = 150    # exclusion radius around detected cross center
POSITION_TOL_MM    = 50     # waypoint considered reached within this distance
APPROACH_OFFSET_MM = 40     # stop this far short of ball (brush sweeps it in)
GOAL_MIN_GAP_MM    = 60     # minimum gap width to count as a goal opening
GOAL_APPROACH_MM   = 220    # approach point distance inside field from goal

OVERSHOOT_COMP_DEG = 5.0
TURN_TIMEOUT_S     = 12.0
ROUTE_INTERVAL_S   = 3.0    # auto-refresh display route while idle
BALL_DRIVE_SPEED   = 10     # slow speed sent to EV3 when sweeping through a ball
REVERSE_THRESHOLD  = 100    # degrees: if angle to NAV > this, reverse is faster

# Waypoint type constants
NAV  = "nav"   # navigation point, no action on arrival
BALL = "ball"  # collect ball on arrival
GOAL = "goal"  # eject balls on arrival


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
# CROSS DETECTION  (position from camera, never hardcoded)
# ════════════════════════════════════════════════════════════════════════════

def detect_cross(world_img_raw, scale=DISPLAY_SCALE):
    """
    Finds the red cross in the central 50% of the world-view image.
    Returns dict with center_mm / center_view / contour, or None.
    """
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


# ════════════════════════════════════════════════════════════════════════════
# GOAL DETECTION  (gaps in the red border = goal openings)
# ════════════════════════════════════════════════════════════════════════════

def _find_gaps(has_red, min_len):
    """Return list of (start, end) index pairs for False runs >= min_len."""
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
    """
    Scans the four border edges of the rectified world image for gaps.
    Gaps wider than GOAL_MIN_GAP_MM are goal openings.
    Returns list of (x_mm, y_mm) goal center positions.
    """
    hsv  = cv2.cvtColor(world_img_raw, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)
    h_i, w_i = mask.shape
    border  = 20
    min_gap = max(3, int(GOAL_MIN_GAP_MM * scale))
    goals   = []

    # Bottom wall  (world y=0,           image rows near bottom)
    strip = mask[max(0, h_i - border):h_i, :]
    for gx0, gx1 in _find_gaps(np.any(strip > 0, axis=0), min_gap):
        goals.append(((gx0 + gx1) / 2.0 / scale, 0.0))

    # Top wall     (world y=BOARD_HEIGHT, image rows near top)
    strip = mask[0:border, :]
    for gx0, gx1 in _find_gaps(np.any(strip > 0, axis=0), min_gap):
        goals.append(((gx0 + gx1) / 2.0 / scale, float(BOARD_HEIGHT_MM)))

    # Left wall    (world x=0,           image cols near left)
    strip = mask[:, 0:border]
    for gy0, gy1 in _find_gaps(np.any(strip > 0, axis=1), min_gap):
        goals.append((0.0, BOARD_HEIGHT_MM - (gy0 + gy1) / 2.0 / scale))

    # Right wall   (world x=BOARD_WIDTH,  image cols near right)
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


def detect_heading(frame):
    """Returns dict with found/heading/center_px, or {found: False}."""
    H, W  = frame.shape[:2]
    f     = max(W, H)
    cam   = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]], dtype=np.float64)
    dist  = np.zeros((5, 1), dtype=np.float64)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _aruco_detector.detectMarkers(gray)
    if ids is None:
        return {"found": False}
    for i, mid in enumerate(ids.flatten()):
        if mid != ROBOT_MARKER_ID:
            continue
        mc = corners[i][0]
        ok, rvec, tvec = cv2.solvePnP(
            _OBJ_PTS, mc.astype(np.float32), cam, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            continue
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
# ROUTE PLANNING
# ════════════════════════════════════════════════════════════════════════════

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seg_dist(p1, p2, pt):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if dx == dy == 0:
        return _dist(p1, pt)
    t = max(0.0, min(1.0,
        ((pt[0]-p1[0])*dx + (pt[1]-p1[1])*dy) / (dx*dx + dy*dy)))
    return _dist((p1[0] + t*dx, p1[1] + t*dy), pt)


def _detour(p1, p2, cross_mm):
    """Single-bypass avoidance around the cross exclusion zone."""
    if _seg_dist(p1, p2, cross_mm) >= CROSS_MARGIN_MM:
        return [p2]
    cx, cy = cross_mm
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    lsq = dx*dx + dy*dy
    if lsq < 1e-9:
        return [p2]
    # Closest point on segment to cross center
    t = max(0.0, min(1.0, ((cx - p1[0])*dx + (cy - p1[1])*dy) / lsq))
    clx, cly = p1[0] + t*dx, p1[1] + t*dy
    # Vector from cross to that closest point
    vx, vy = clx - cx, cly - cy
    mag = math.hypot(vx, vy)
    if mag < 1e-9:
        # Cross sits exactly on path — push perpendicular to path direction
        pl = math.hypot(dx, dy)
        vx, vy, mag = -dy / pl, dx / pl, 1.0
    push   = CROSS_MARGIN_MM * 1.8
    bypass = (cx + vx / mag * push, cy + vy / mag * push)
    return [bypass, p2]


def _wall_approach(ball_mm):
    """Returns perpendicular approach point for wall balls, or None."""
    x, y   = ball_mm
    apd    = WALL_APPROACH_MM
    margin = WALL_MARGIN_MM
    near_L = x                   < margin
    near_R = BOARD_WIDTH_MM  - x < margin
    near_B = y                   < margin
    near_T = BOARD_HEIGHT_MM - y < margin

    if near_L and near_B: return (x + apd, y + apd)
    if near_R and near_B: return (x - apd, y + apd)
    if near_L and near_T: return (x + apd, y - apd)
    if near_R and near_T: return (x - apd, y - apd)
    if near_L:            return (x + apd, y)
    if near_R:            return (x - apd, y)
    if near_B:            return (x, y + apd)
    if near_T:            return (x, y - apd)
    return None


def _goal_approach(goal_mm):
    """Returns a point GOAL_APPROACH_MM inside the field from the goal wall."""
    x, y = goal_mm
    d = GOAL_APPROACH_MM
    if x < 10:                       return (d, y)
    if x > BOARD_WIDTH_MM  - 10:     return (BOARD_WIDTH_MM  - d, y)
    if y < 10:                       return (x, d)
    if y > BOARD_HEIGHT_MM - 10:     return (x, BOARD_HEIGHT_MM - d)
    return None


def _add_waypoints(route, current, target, cross_mm, wp_type):
    """Add cross-avoiding path from current to target. Returns new current."""
    path = _detour(current, target, cross_mm)
    for wp in path[:-1]:
        route.append((wp[0], wp[1], NAV))
    route.append((target[0], target[1], wp_type))
    return target


def plan_route(white_mm, orange_mm, robot_mm, cross_mm, goals_mm):
    """
    Waypoint order:
      1. White balls (nearest-neighbor, wall approach, cross avoidance)
      2. Orange ball(s) last
      3. When no balls remain: nearest goal delivery

    Cross and goal positions come from camera detection, never hardcoded.
    """
    route   = []
    current = robot_mm

    def add_ball(ball):
        nonlocal current
        approach = _wall_approach(ball)
        if approach:
            current = _add_waypoints(route, current, approach, cross_mm, NAV)
        current = _add_waypoints(route, current, ball, cross_mm, BALL)

    # White balls: nearest-neighbor
    remaining = list(white_mm)
    while remaining:
        nxt = min(remaining, key=lambda p: _dist(current, p))
        remaining.remove(nxt)
        add_ball(nxt)

    # Orange ball(s) always last
    for nxt in orange_mm:
        add_ball(nxt)

    # Goal delivery when no balls remain
    if not white_mm and not orange_mm and goals_mm:
        goal     = min(goals_mm, key=lambda g: _dist(current, g))
        approach = _goal_approach(goal)
        if approach:
            current = _add_waypoints(route, current, approach, cross_mm, NAV)
        _add_waypoints(route, current, goal, cross_mm, GOAL)

    return route


# ════════════════════════════════════════════════════════════════════════════
# DRAWING
# ════════════════════════════════════════════════════════════════════════════

def draw_route(world_img, route, start_mm, scale=DISPLAY_SCALE):
    if not route:
        return
    color_map = {NAV: (80, 180, 80), BALL: (0, 255, 255), GOAL: (0, 140, 255)}
    pts = [world_to_view(start_mm, scale)] + \
          [world_to_view((p[0], p[1]), scale) for p in route]
    for i in range(len(pts) - 1):
        col = color_map.get(route[i][2], (0, 255, 0))
        cv2.line(world_img, pts[i], pts[i+1], col, 2)
        ang = math.atan2(pts[i+1][1]-pts[i][1], pts[i+1][0]-pts[i][0])
        mx  = (pts[i][0] + pts[i+1][0]) // 2
        my  = (pts[i][1] + pts[i+1][1]) // 2
        for sign in (-0.4, 0.4):
            ex = int(mx - 7*math.cos(ang - sign))
            ey = int(my - 7*math.sin(ang - sign))
            cv2.line(world_img, (mx, my), (ex, ey), col, 2)


def draw_robot(world_img, dir_info, H_px_to_world, scale=DISPLAY_SCALE):
    if not dir_info.get("found"):
        return
    cx, cy = dir_info["center_px"]
    wmm    = pixel_to_world((cx, cy), H_px_to_world)
    pv     = world_to_view(wmm, scale)
    h      = dir_info["heading"]
    cv2.circle(world_img, pv, 10, (255, 180, 0), -1)
    ex = int(pv[0] + 25*math.cos(math.radians(h)))
    ey = int(pv[1] - 25*math.sin(math.radians(h)))
    cv2.arrowedLine(world_img, pv, (ex, ey), (255, 180, 0), 2, tipLength=0.4)


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
        cv2.putText(world_img, "GOAL", (pv[0]+5, pv[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 100), 1)


# ════════════════════════════════════════════════════════════════════════════
# CAMERA + SHARED STATE
# ════════════════════════════════════════════════════════════════════════════

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise SystemExit("Cannot open camera {}".format(CAMERA_INDEX))

cfg          = DetectorConfig()
prev_corners = None

last_H_px_world = None
last_cross      = None
last_goals      = []
last_dir_info   = {"found": False}
last_whites_px  = []
last_oranges_px = []
last_route      = []
start_mm        = (BOARD_WIDTH_MM * 0.1, BOARD_HEIGHT_MM * 0.5)
last_route_time = 0.0


# ════════════════════════════════════════════════════════════════════════════
# ROBOT EXECUTOR THREAD
# ════════════════════════════════════════════════════════════════════════════

robot_running  = threading.Event()
robot_stop_req = threading.Event()
current_route  = []
route_lock     = threading.Lock()


def _replan_from_camera():
    """
    Recomputes the route from current camera detections and robot position.
    Waits up to 3s for homography if temporarily unavailable.
    Updates current_route and last_route (display). Stops the robot if no
    balls and no goal remain.
    """
    global last_route

    wait_end = time.time() + 3.0
    while last_H_px_world is None and time.time() < wait_end:
        time.sleep(0.1)

    if last_H_px_world is None:
        print("[robot] No homography - cannot replan, stopping")
        robot_running.clear()
        return

    cross_mm = (last_cross["center_mm"] if last_cross
                else (BOARD_WIDTH_MM / 2.0, BOARD_HEIGHT_MM / 2.0))
    w_mm = [pixel_to_world((x, y), last_H_px_world) for (x, y, r) in last_whites_px]
    o_mm = [pixel_to_world((x, y), last_H_px_world) for (x, y, r) in last_oranges_px]
    new_route = plan_route(w_mm, o_mm, start_mm, cross_mm, last_goals)
    last_route = new_route
    with route_lock:
        current_route.clear()
        current_route.extend(new_route)

    if new_route:
        print("[robot] Replanned: {} waypoints ({}W {}O)".format(
            len(new_route), len(w_mm), len(o_mm)))
    else:
        print("[robot] All balls collected - heading to goal")
        robot.stop()
        robot_running.clear()


def robot_executor():
    while True:
        robot_running.wait()

        with route_lock:
            if not current_route:
                robot_running.clear()
                print("[robot] Route complete")
                continue
            wp = current_route.pop(0)

        if robot_stop_req.is_set():
            robot_running.clear()
            robot_stop_req.clear()
            continue

        pos     = start_mm
        wp_pos  = (wp[0], wp[1])
        wp_type = wp[2]

        # Live cross avoidance: re-check from the robot's actual current position.
        # Only for BALL/GOAL waypoints so the bypass NAV we insert never loops.
        if wp_type != NAV and last_cross is not None:
            detour_pts = _detour(pos, wp_pos, last_cross["center_mm"])
            if len(detour_pts) > 1:
                bypass_pt = detour_pts[0]
                print("[robot] Cross in path -> bypass ({:.0f},{:.0f})".format(
                    bypass_pt[0], bypass_pt[1]))
                with route_lock:
                    current_route.insert(0, wp)  # re-queue original
                wp      = (bypass_pt[0], bypass_pt[1], NAV)
                wp_pos  = bypass_pt
                wp_type = NAV

        print("[robot] -> {} ({})".format(wp_pos, wp_type))

        dx   = wp_pos[0] - pos[0]
        dy   = wp_pos[1] - pos[1]
        dist = math.hypot(dx, dy)

        if dist >= POSITION_TOL_MM:
            # 1. Camera-guided turn
            # For NAV waypoints: if the ball is behind us, reverse is faster.
            # Flip the target heading 180° and drive negative mm instead.
            target_h   = math.degrees(math.atan2(dy, dx))
            drive_sign = 1   # +1 forward, -1 reverse
            if wp_type == NAV:
                cur_h = last_dir_info.get("heading")
                if cur_h is not None:
                    diff = (target_h - cur_h + 180) % 360 - 180
                    if abs(diff) > REVERSE_THRESHOLD:
                        target_h   = (target_h + 180) % 360
                        drive_sign = -1

            ok = robot.turn_to_heading(
                target_h,
                lambda: last_dir_info.get("heading"),
                overshoot_comp=OVERSHOOT_COMP_DEG,
                timeout=TURN_TIMEOUT_S,
                stop_fn=lambda: robot_stop_req.is_set(),
            )
            if robot_stop_req.is_set():
                robot_running.clear()
                robot_stop_req.clear()
                continue
            if not ok:
                with route_lock:
                    current_route.insert(0, wp)
                time.sleep(0.5)
                continue

            # 2. Camera-guided drive: start motors, stop when ArUco reaches wp
            # For BALL: stop slightly before the ball so collector can sweep it
            tol       = APPROACH_OFFSET_MM if wp_type == BALL else POSITION_TOL_MM
            print("[robot] Driving {} to ({:.0f},{:.0f}) tol={}mm".format(
                "rev" if drive_sign < 0 else "fwd",
                wp_pos[0], wp_pos[1], tol))
            reached = robot.drive_to_position(
                wp_pos,
                lambda: start_mm,
                reverse=(drive_sign < 0),
                tol_mm=tol,
                timeout=15.0,
                stop_fn=lambda: robot_stop_req.is_set(),
            )
            if not reached:
                print("[robot] Did not reach ({:.0f},{:.0f}) - continuing".format(
                    wp_pos[0], wp_pos[1]))

        if robot_stop_req.is_set():
            robot_running.clear()
            robot_stop_req.clear()
            continue

        # 3. Arrival action
        if wp_type == NAV:
            robot.motor_stop()
            time.sleep(0.2)
            # Check route: is the next BALL waypoint still backed by a real ball?
            with route_lock:
                next_wp = current_route[0] if current_route else None
            if next_wp is not None and next_wp[2] == BALL and last_H_px_world is not None:
                target = (next_wp[0], next_wp[1])
                all_balls = (
                    [pixel_to_world((x, y), last_H_px_world) for (x, y, r) in last_whites_px] +
                    [pixel_to_world((x, y), last_H_px_world) for (x, y, r) in last_oranges_px]
                )
                if not any(_dist(target, b) < 80 for b in all_balls):
                    print("[robot] Ball at next waypoint gone - replanning")
                    _replan_from_camera()
                    continue
            with route_lock:
                remaining = len(current_route)
            print("[robot] NAV reached - {} waypoints left".format(remaining))

        elif wp_type == BALL:
            # Wait for the collected ball to leave the camera frame
            time.sleep(1.5)
            _replan_from_camera()

        elif wp_type == GOAL:
            print("[robot] Ejecting balls into goal...")
            robot.eject()
            print("[robot] Done!")
            robot_running.clear()


threading.Thread(target=robot_executor, daemon=True).start()


# ════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════

for win in ["camera", "world", "edges"]:
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

print("Keys:  r=route  g=go  s=stop  c=collect  e=eject  p=status  ESC=quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # ── 1. Board detection ──────────────────────────────────────────────────
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)
    bp   = extract_boundary_points(mask)
    mdl  = fit_frame_lines(bp)

    draw_model_line(display, mdl["top"],    (255,   0,   0), 2)
    draw_model_line(display, mdl["bottom"], (  0,   0, 255), 2)
    draw_model_line(display, mdl["left"],   (  0, 255,   0), 2)
    draw_model_line(display, mdl["right"],  (  0, 255, 255), 2)

    disp_h = int(BOARD_HEIGHT_MM * DISPLAY_SCALE)
    disp_w = int(BOARD_WIDTH_MM  * DISPLAY_SCALE)
    world_img = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)

    if all(mdl[k] is not None for k in ["top", "bottom", "left", "right"]):
        new_corners = {
            "TL": intersect_horizontal_vertical(mdl["top"],    mdl["left"]),
            "TR": intersect_horizontal_vertical(mdl["top"],    mdl["right"]),
            "BL": intersect_horizontal_vertical(mdl["bottom"], mdl["left"]),
            "BR": intersect_horizontal_vertical(mdl["bottom"], mdl["right"]),
        }
        if all(v is not None for v in new_corners.values()):
            if prev_corners is None:
                corners = new_corners
            else:
                corners = {n: (
                    int(SMOOTH_ALPHA * prev_corners[n][0] +
                        (1 - SMOOTH_ALPHA) * new_corners[n][0]),
                    int(SMOOTH_ALPHA * prev_corners[n][1] +
                        (1 - SMOOTH_ALPHA) * new_corners[n][1]),
                ) for n in new_corners}
            prev_corners = corners

            H_px_to_world, H_px_to_view, dw, dh = compute_homographies(
                corners, DISPLAY_SCALE)
            last_H_px_world = H_px_to_world

            world_raw = cv2.warpPerspective(frame, H_px_to_view, (dw, dh))
            world_img = world_raw.copy()
            draw_world_grid(world_img, DISPLAY_SCALE, step_mm=200)

            # ── 2. Cross ────────────────────────────────────────────────────
            cross = detect_cross(world_raw, DISPLAY_SCALE)
            if cross:
                last_cross = cross
            if last_cross:
                cv2.drawContours(world_img, [last_cross["contour"]], -1,
                                 (255, 255, 255), 2)
                cv2.circle(world_img, last_cross["center_view"], 6, (0, 0, 255), -1)

            # ── 3. Goals ────────────────────────────────────────────────────
            goals = detect_goals(world_raw, DISPLAY_SCALE)
            if goals:
                last_goals = goals
            draw_goals(world_img, last_goals)

    # ── 4. Balls ────────────────────────────────────────────────────────────
    whites_px, oranges_px, edges = detect_balls(frame, cfg)
    last_whites_px  = whites_px
    last_oranges_px = oranges_px

    for (x, y, r) in whites_px:
        cv2.circle(display, (x, y), r, (0, 255, 255), 2)
    for (x, y, r) in oranges_px:
        cv2.circle(display, (x, y), r, (0, 128, 255), 2)

    if last_H_px_world is not None:
        draw_balls(world_img, whites_px, oranges_px, last_H_px_world)

    # ── 5. Heading ──────────────────────────────────────────────────────────
    dir_info = detect_heading(frame)
    if dir_info["found"]:
        last_dir_info = dir_info
        if last_H_px_world is not None:
            start_mm = pixel_to_world(dir_info["center_px"], last_H_px_world)

    if last_H_px_world is not None:
        draw_robot(world_img, last_dir_info, last_H_px_world)

    h_str = "{:+.1f}".format(last_dir_info["heading"]) \
            if last_dir_info.get("found") else "?"
    cv2.putText(display,
        "Heading:{}  Balls:{}W {}O  Goals:{}  Cross:{}  Run:{}".format(
            h_str, len(whites_px), len(oranges_px),
            len(last_goals),
            "OK" if last_cross else "?",
            "YES" if robot_running.is_set() else "no",
        ), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # ── 6. Route display (auto-refresh every 3 s while idle) ────────────────
    now = time.time()
    if (now - last_route_time >= ROUTE_INTERVAL_S
            and last_H_px_world is not None
            and last_cross is not None
            and not robot_running.is_set()):
        cross_mm = last_cross["center_mm"]
        w_mm = [pixel_to_world((x, y), last_H_px_world) for (x, y, r) in whites_px]
        o_mm = [pixel_to_world((x, y), last_H_px_world) for (x, y, r) in oranges_px]
        last_route = plan_route(w_mm, o_mm, start_mm, cross_mm, last_goals)
        last_route_time = now

    draw_route(world_img, last_route, start_mm)

    cv2.imshow("camera", display)
    cv2.imshow("world",  world_img)
    cv2.imshow("edges",  edges)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        if robot_running.is_set():
            print("Robot running - press s first")
        elif last_H_px_world is None or last_cross is None:
            print("Waiting for board + cross detection")
        else:
            cross_mm = last_cross["center_mm"]
            w_mm = [pixel_to_world((x, y), last_H_px_world) for (x, y, r) in whites_px]
            o_mm = [pixel_to_world((x, y), last_H_px_world) for (x, y, r) in oranges_px]
            last_route = plan_route(w_mm, o_mm, start_mm, cross_mm, last_goals)
            with route_lock:
                current_route.clear()
                current_route.extend(last_route)
            print("Route: {} waypoints - press g to start".format(len(last_route)))

    elif key == ord("g"):
        if not robot.connected:
            print("Robot not connected")
        elif not current_route:
            print("No route - press r first")
        else:
            robot_stop_req.clear()
            robot_running.set()
            print("GO")

    elif key == ord("s"):
        robot_stop_req.set()
        robot_running.clear()
        robot.stop()
        print("STOP")

    elif key == ord("c"):
        robot.collect()
        print("Collect")

    elif key == ord("e"):
        robot.eject()
        print("Eject")

    elif key == ord("p"):
        print("Heading:{}  Balls:{}W {}O  Goals:{}  Cross:{}  Route:{} wp  Run:{}".format(
            h_str, len(whites_px), len(oranges_px),
            len(last_goals),
            "OK" if last_cross else "?",
            len(current_route),
            robot_running.is_set(),
        ))

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
