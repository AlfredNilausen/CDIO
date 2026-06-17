
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_bestfit"))

import cv2
import math
import numpy as np

from homography import (
    compute_homographies,
    pixel_to_world,
    world_to_view,
    draw_world_grid,
    BOARD_WIDTH_MM,
    BOARD_HEIGHT_MM,
)

from geometry import intersect_horizontal_vertical

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

DISPLAY_SCALE = 0.5
CAMERA_INDEX = 1

# Green detection
GREEN_H_MIN = 25
GREEN_H_MAX = 95
GREEN_S_MIN = 50
GREEN_V_MIN = 50
MIN_BLOB_AREA = 60

# Robot geometry (green markers form a rectangle)
ROBOT_WIDTH_MM = 185.0
ROBOT_LENGTH_MM = 280.0
HALF_W = ROBOT_WIDTH_MM / 2.0
LENGTH = ROBOT_LENGTH_MM

# Hvis FL/FR byttes rundt i overlay, skift til -1.0
SIDE_SIGN = 1.0

# ArUco config
ROBOT_MARKER_ID = 0
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# På dit seneste screenshot var +180 forkert, så vi sætter 0 her.
# Hvis robotpilen stadig vender forkert, prøv 180.0 eller +/-90.0
ARUCO_HEADING_OFFSET_DEG = 0.0

# History fallback
MAX_HISTORY_ASSIGN_MM = 220.0

# ------------------------------------------------------------------
# IMPORT FROM EXISTING PROJECT
# ------------------------------------------------------------------

from red_mask import red_mask
from boundary_points import extract_boundary_points
from best_fit_lines import fit_frame_lines

# ------------------------------------------------------------------
# ROBOT MODEL
# Local frame:
#   origin = TOP CENTER (midpoint between FL and FR)
#   +x     = forward direction
#   +y     = sideways
#
# Front row = x = 0
# Back row  = x = -ROBOT_LENGTH_MM
# ------------------------------------------------------------------

MODEL = {
    "FL": (0.0,      -HALF_W),
    "FR": (0.0,       HALF_W),
    "BL": (-LENGTH,  -HALF_W),
    "BR": (-LENGTH,   HALF_W),
}

# ------------------------------------------------------------------
# ARUCO SETUP
# ------------------------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

MARKER_SIZE_MM = 80.0
_half = MARKER_SIZE_MM / 2.0

OBJ_PTS = np.array([
    [-_half,  _half, 0],
    [ _half,  _half, 0],
    [ _half, -_half, 0],
    [-_half, -_half, 0],
], dtype=np.float32)

# ------------------------------------------------------------------
# STATE / HISTORY
# ------------------------------------------------------------------

LAST_TOP_CENTER = None
LAST_SOLVED_MARKERS = None
LAST_HEADING = None

# ------------------------------------------------------------------
# VECTOR HELPERS
# ------------------------------------------------------------------

def add2(a, b):
    return (a[0] + b[0], a[1] + b[1])


def sub2(a, b):
    return (a[0] - b[0], a[1] - b[1])


def mul2(v, k):
    return (v[0] * k, v[1] * k)


def dot2(a, b):
    return a[0] * b[0] + a[1] * b[1]


def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def norm2(v):
    n = math.hypot(v[0], v[1])
    if n < 1e-9:
        return (0.0, 0.0)
    return (v[0] / n, v[1] / n)


def midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def heading_unit(heading_deg):
    a = math.radians(heading_deg)
    return (math.cos(a), math.sin(a))


def side_unit_from_heading(heading_deg):
    f = heading_unit(heading_deg)
    return (SIDE_SIGN * (-f[1]), SIDE_SIGN * f[0])

# ------------------------------------------------------------------
# MODEL / WORLD CONVERSIONS
# ------------------------------------------------------------------

def world_from_top(top_center, label, heading_deg):
    f = heading_unit(heading_deg)
    s = side_unit_from_heading(heading_deg)

    xf, ys = MODEL[label]
    return (
        top_center[0] + xf * f[0] + ys * s[0],
        top_center[1] + xf * f[1] + ys * s[1],
    )


def top_from_known_marker(pt_world, label, heading_deg):
    f = heading_unit(heading_deg)
    s = side_unit_from_heading(heading_deg)

    xf, ys = MODEL[label]
    return (
        pt_world[0] - xf * f[0] - ys * s[0],
        pt_world[1] - xf * f[1] - ys * s[1],
    )


def solve_from_any_labeled_points(labels_found, heading_deg):
    """
    Compute top_center from any subset of correctly labeled FL/FR/BL/BR points,
    then reconstruct all 4 markers from the model.
    """
    usable = {k: v for k, v in labels_found.items() if k in MODEL}
    if not usable:
        return None, None

    top_candidates = [top_from_known_marker(pt, label, heading_deg)
                      for label, pt in usable.items()]

    tx = sum(p[0] for p in top_candidates) / len(top_candidates)
    ty = sum(p[1] for p in top_candidates) / len(top_candidates)
    top_center = (tx, ty)

    solved = {}
    for label in MODEL.keys():
        if label in usable:
            solved[label] = usable[label]
        else:
            solved[label] = world_from_top(top_center, label, heading_deg)

    return solved, top_center


def compute_front_normal(fl, fr, heading_deg):
    """Normal on FL-FR edge, chosen to point in heading direction."""
    edge = (fr[0] - fl[0], fr[1] - fl[1])
    n1 = norm2((-edge[1], edge[0]))
    n2 = (-n1[0], -n1[1])
    h = heading_unit(heading_deg)
    return n2 if dot2(n2, h) > dot2(n1, h) else n1

# ------------------------------------------------------------------
# ARUCO HEADING ONLY (position only used for side/front classification)
# ------------------------------------------------------------------

def detect_heading(frame):
    H, W = frame.shape[:2]
    f = max(H, W)

    cam = np.array([
        [f, 0, W / 2],
        [0, f, H / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist = np.zeros((5, 1))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is None:
        return None

    for i, mid in enumerate(ids.flatten()):
        if mid != ROBOT_MARKER_ID:
            continue

        mc = corners[i][0]

        ok, rvec, tvec = cv2.solvePnP(
            OBJ_PTS,
            mc.astype(np.float32),
            cam,
            dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            continue

        R, _ = cv2.Rodrigues(rvec)
        heading = math.degrees(math.atan2(-R[1, 0], R[0, 0]))
        heading = (heading + ARUCO_HEADING_OFFSET_DEG) % 360.0

        center = (
            int(mc[:, 0].mean()),
            int(mc[:, 1].mean())
        )

        return {
            "heading": heading,
            "center_px": center
        }

    return None

# ------------------------------------------------------------------
# GREEN BLOBS
# ------------------------------------------------------------------

def detect_green_blobs(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        (GREEN_H_MIN, GREEN_S_MIN, GREEN_V_MIN),
        (GREEN_H_MAX, 255, 255)
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_BLOB_AREA:
            continue

        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-6:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        blobs.append((cx, cy))

    return blobs, mask

# ------------------------------------------------------------------
# CLASSIFICATION USING:
#   - green points in WORLD coordinates
#   - ArUCo heading for front/back
#   - ArUCo center (world) only to decide which side of robot a point/side lies on
# ------------------------------------------------------------------

def classify_visible_blobs(blobs_world, heading_deg, aruco_center_world):
    """
    Returns:
        labels_found: subset of {FL, FR, BL, BR}
        case_name: e.g. 4pts, 3pts_frontpair, 2pts_side, 1pt
    """
    f = heading_unit(heading_deg)
    s = side_unit_from_heading(heading_deg)

    pts = []
    for p in blobs_world:
        pts.append({
            "pt": p,
            "pf": dot2(p, f),
            "ps": dot2(p, s),
        })

    n = len(pts)
    labels = {}

    if n == 0:
        return {}, "none"

    # --------------------------------------------------------------
    # 4 visible points
    # --------------------------------------------------------------
    if n == 4:
        pts_sorted = sorted(pts, key=lambda t: t["pf"], reverse=True)
        front = pts_sorted[:2]
        back = pts_sorted[2:]

        front = sorted(front, key=lambda t: t["ps"])
        back = sorted(back, key=lambda t: t["ps"])

        labels["FL"] = front[0]["pt"]
        labels["FR"] = front[1]["pt"]
        labels["BL"] = back[0]["pt"]
        labels["BR"] = back[1]["pt"]
        return labels, "4pts"

    # --------------------------------------------------------------
    # 3 visible points
    # --------------------------------------------------------------
    if n == 3:
        pts_sorted = sorted(pts, key=lambda t: t["pf"], reverse=True)

        pf0 = pts_sorted[0]["pf"]
        pf1 = pts_sorted[1]["pf"]
        pf2 = pts_sorted[2]["pf"]

        span_front = abs(pf0 - pf1)
        span_back = abs(pf1 - pf2)

        # Two front points + one back point
        if span_front < span_back:
            front = sorted(pts_sorted[:2], key=lambda t: t["ps"])
            back_one = pts_sorted[2]

            labels["FL"] = front[0]["pt"]
            labels["FR"] = front[1]["pt"]

            rel = (
                back_one["pt"][0] - aruco_center_world[0],
                back_one["pt"][1] - aruco_center_world[1],
            )
            if dot2(rel, s) < 0:
                labels["BL"] = back_one["pt"]
            else:
                labels["BR"] = back_one["pt"]

            return labels, "3pts_frontpair"

        # One front point + two back points
        else:
            front_one = pts_sorted[0]
            back = sorted(pts_sorted[1:], key=lambda t: t["ps"])

            labels["BL"] = back[0]["pt"]
            labels["BR"] = back[1]["pt"]

            rel = (
                front_one["pt"][0] - aruco_center_world[0],
                front_one["pt"][1] - aruco_center_world[1],
            )
            if dot2(rel, s) < 0:
                labels["FL"] = front_one["pt"]
            else:
                labels["FR"] = front_one["pt"]

            return labels, "3pts_backpair"

    # --------------------------------------------------------------
    # 2 visible points
    # --------------------------------------------------------------
    if n == 2:
        p1, p2 = pts[0], pts[1]
        d = dist2(p1["pt"], p2["pt"])
        diag = math.hypot(ROBOT_WIDTH_MM, ROBOT_LENGTH_MM)

        # Same row (front row or back row): distance ~ 180 mm
        if abs(d - ROBOT_WIDTH_MM) < 45:
            pair = sorted(pts, key=lambda t: t["ps"])
            pair_mid = midpoint(pair[0]["pt"], pair[1]["pt"])
            rel = (
                pair_mid[0] - aruco_center_world[0],
                pair_mid[1] - aruco_center_world[1],
            )
            row_forwardness = dot2(rel, f)

            if row_forwardness >= 0:
                labels["FL"] = pair[0]["pt"]
                labels["FR"] = pair[1]["pt"]
                return labels, "2pts_frontrow"
            else:
                labels["BL"] = pair[0]["pt"]
                labels["BR"] = pair[1]["pt"]
                return labels, "2pts_backrow"

        # Same side (left side or right side): distance ~ 280 mm
        if abs(d - ROBOT_LENGTH_MM) < 50:
            pair = sorted(pts, key=lambda t: t["pf"], reverse=True)
            front = pair[0]
            back = pair[1]

            pair_mid = midpoint(front["pt"], back["pt"])
            rel = (
                pair_mid[0] - aruco_center_world[0],
                pair_mid[1] - aruco_center_world[1],
            )

            if dot2(rel, s) < 0:
                labels["FL"] = front["pt"]
                labels["BL"] = back["pt"]
            else:
                labels["FR"] = front["pt"]
                labels["BR"] = back["pt"]

            return labels, "2pts_side"

        # Diagonal: distance ~ 333 mm
        if abs(d - diag) < 60:
            pair = sorted(pts, key=lambda t: t["pf"], reverse=True)
            front = pair[0]
            back = pair[1]

            rel_front = (
                front["pt"][0] - aruco_center_world[0],
                front["pt"][1] - aruco_center_world[1],
            )

            if dot2(rel_front, s) < 0:
                labels["FL"] = front["pt"]
                labels["BR"] = back["pt"]
            else:
                labels["FR"] = front["pt"]
                labels["BL"] = back["pt"]

            return labels, "2pts_diag"

        return {}, "2pts_unknown"

    # --------------------------------------------------------------
    # 1 visible point
    # --------------------------------------------------------------
    if n == 1:
        p = pts[0]["pt"]
        rel = (
            p[0] - aruco_center_world[0],
            p[1] - aruco_center_world[1],
        )
        is_front = dot2(rel, f) >= 0
        is_left = dot2(rel, s) < 0

        if is_front and is_left:
            labels["FL"] = p
        elif is_front and not is_left:
            labels["FR"] = p
        elif (not is_front) and is_left:
            labels["BL"] = p
        else:
            labels["BR"] = p

        return labels, "1pt"

    return {}, "unknown"

# ------------------------------------------------------------------
# HISTORY-BASED FALLBACK
# ------------------------------------------------------------------

def assign_using_last_pose(blobs_world, last_solved_markers, max_dist=MAX_HISTORY_ASSIGN_MM):
    """Assign observed world-space blobs to nearest previous FL/FR/BL/BR."""
    if last_solved_markers is None:
        return {}

    labels_found = {}
    used_labels = set()

    for p in blobs_world:
        best_lab = None
        best_d = float("inf")

        for lab, prev_pt in last_solved_markers.items():
            if lab in used_labels:
                continue
            d = dist2(p, prev_pt)
            if d < best_d:
                best_d = d
                best_lab = lab

        if best_lab is not None and best_d <= max_dist:
            labels_found[best_lab] = p
            used_labels.add(best_lab)

    return labels_found

# ------------------------------------------------------------------
# SOLVE ROBOT FROM LABELS
# ------------------------------------------------------------------

def solve_robot_from_labels(labels_found, case_name, heading_deg,
                            last_top_center=None, last_solved_markers=None):
    # Direct solve from any correctly labeled subset
    if case_name in (
        "4pts",
        "3pts_frontpair",
        "3pts_backpair",
        "2pts_frontrow",
        "2pts_backrow",
        "2pts_side",
        "2pts_diag",
        "1pt",
    ):
        solved, top_center = solve_from_any_labeled_points(labels_found, heading_deg)
        return solved, top_center

    # Unknown shapes -> try history labels if any were assigned later in caller
    if any(k in MODEL for k in labels_found.keys()):
        solved, top_center = solve_from_any_labeled_points(labels_found, heading_deg)
        return solved, top_center

    return None, None

# ------------------------------------------------------------------
# DRAWING
# ------------------------------------------------------------------

def draw_robot_solution(world_img, solved_markers, top_center, heading_deg,
                        display_scale, case_name, measured_labels=None):
    if measured_labels is None:
        measured_labels = set(solved_markers.keys())

    color_measured = {
        "FL": (0, 255, 255),
        "FR": (255, 255, 0),
        "BL": (255, 0, 255),
        "BR": (0, 165, 255),
    }
    color_estimated = {
        "FL": (0, 180, 180),
        "FR": (180, 180, 0),
        "BL": (180, 0, 180),
        "BR": (0, 110, 180),
    }

    # Draw markers and labels
    for label in ["FL", "FR", "BL", "BR"]:
        if label not in solved_markers:
            continue

        p_view = world_to_view(solved_markers[label], display_scale)
        measured = label in measured_labels
        color = color_measured[label] if measured else color_estimated[label]
        radius = 7 if measured else 5
        thickness = -1 if measured else 2
        cv2.circle(world_img, p_view, radius, color, thickness)
        suffix = "" if measured else "*"
        cv2.putText(
            world_img,
            f"{label}{suffix}",
            (p_view[0] + 8, p_view[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    fl = solved_markers["FL"]
    fr = solved_markers["FR"]

    # Front edge
    cv2.line(
        world_img,
        world_to_view(fl, display_scale),
        world_to_view(fr, display_scale),
        (0, 255, 0),
        2
    )

    # Top center
    top_view = world_to_view(top_center, display_scale)
    cv2.circle(world_img, top_view, 8, (255, 0, 0), -1)

    # Normal on the front edge -> robot forward direction
    forward_n = compute_front_normal(fl, fr, heading_deg)
    arrow_len = 120.0
    ex = top_center[0] + arrow_len * forward_n[0]
    ey = top_center[1] + arrow_len * forward_n[1]

    cv2.arrowedLine(
        world_img,
        world_to_view(top_center, display_scale),
        world_to_view((ex, ey), display_scale),
        (0, 0, 255),
        3
    )

    # Geometric center (optional)
    geom_center = (
        (solved_markers["FL"][0] + solved_markers["FR"][0] +
         solved_markers["BL"][0] + solved_markers["BR"][0]) / 4.0,
        (solved_markers["FL"][1] + solved_markers["FR"][1] +
         solved_markers["BL"][1] + solved_markers["BR"][1]) / 4.0,
    )
    cv2.circle(world_img, world_to_view(geom_center, display_scale), 5, (200, 200, 200), -1)

    cv2.putText(
        world_img,
        "Top X={:.0f} Y={:.0f} H={:.1f} {}".format(
            top_center[0],
            top_center[1],
            heading_deg,
            case_name
        ),
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    global LAST_TOP_CENTER, LAST_SOLVED_MARKERS, LAST_HEADING

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open camera index:", CAMERA_INDEX)
        sys.exit(1)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        display = frame.copy()

        # ----------------------------------------------------------
        # Board detection
        # ----------------------------------------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_red = red_mask(hsv)

        bp = extract_boundary_points(mask_red)
        mdl = fit_frame_lines(bp)

        required = ["top", "bottom", "left", "right"]
        if not all(mdl[k] is not None for k in required):
            cv2.imshow("camera", display)
            if cv2.waitKey(1) == 27:
                break
            continue

        corners = {
            "TL": intersect_horizontal_vertical(mdl["top"], mdl["left"]),
            "TR": intersect_horizontal_vertical(mdl["top"], mdl["right"]),
            "BL": intersect_horizontal_vertical(mdl["bottom"], mdl["left"]),
            "BR": intersect_horizontal_vertical(mdl["bottom"], mdl["right"]),
        }

        if any(v is None for v in corners.values()):
            cv2.imshow("camera", display)
            if cv2.waitKey(1) == 27:
                break
            continue

        H_px_to_world, H_px_to_view, disp_w, disp_h = compute_homographies(
            corners,
            DISPLAY_SCALE
        )

        world_img = cv2.warpPerspective(frame, H_px_to_view, (disp_w, disp_h))
        draw_world_grid(world_img, DISPLAY_SCALE)

        # ----------------------------------------------------------
        # ArUCo heading
        # ----------------------------------------------------------
        aruco_pose = detect_heading(frame)
        if aruco_pose is None:
            cv2.imshow("camera", display)
            cv2.imshow("world", world_img)
            if cv2.waitKey(1) == 27:
                break
            continue

        heading = aruco_pose["heading"]
        LAST_HEADING = heading
        aruco_center_world = pixel_to_world(aruco_pose["center_px"], H_px_to_world)

        # ----------------------------------------------------------
        # Green blobs
        # ----------------------------------------------------------
        blobs_px, green_mask = detect_green_blobs(frame)
        for p in blobs_px:
            cv2.circle(display, p, 10, (0, 255, 0), 2)

        blobs_world = [pixel_to_world(p, H_px_to_world) for p in blobs_px]

        # ----------------------------------------------------------
        # First-pass classification
        # ----------------------------------------------------------
        labels_found, case_name = classify_visible_blobs(
            blobs_world,
            heading,
            aruco_center_world
        )

        measured_labels = set(k for k in labels_found.keys() if k in MODEL)

        solved_markers, top_center = solve_robot_from_labels(
            labels_found,
            case_name,
            heading,
            last_top_center=LAST_TOP_CENTER,
            last_solved_markers=LAST_SOLVED_MARKERS
        )

        # ----------------------------------------------------------
        # History fallback for difficult/unknown frames
        # ----------------------------------------------------------
        if (solved_markers is None or top_center is None) and LAST_SOLVED_MARKERS is not None and len(blobs_world) > 0:
            hist_labels = assign_using_last_pose(blobs_world, LAST_SOLVED_MARKERS)
            if hist_labels:
                solved_markers, top_center = solve_from_any_labeled_points(hist_labels, heading)
                measured_labels = set(hist_labels.keys())
                case_name = "history"

        # ----------------------------------------------------------
        # Draw + save state
        # ----------------------------------------------------------
        if solved_markers is not None and top_center is not None:
            LAST_TOP_CENTER = top_center
            LAST_SOLVED_MARKERS = dict(solved_markers)

            draw_robot_solution(
                world_img,
                solved_markers,
                top_center,
                heading,
                DISPLAY_SCALE,
                case_name,
                measured_labels=measured_labels
            )

            print(
                "\rTop center: ({:.0f}, {:.0f}) heading={:.1f} case={}".format(
                    top_center[0],
                    top_center[1],
                    heading,
                    case_name
                ),
                end=""
            )
        else:
            cv2.putText(
                world_img,
                "Could not solve pose (need more points/history)",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow("camera", display)
        cv2.imshow("green mask", green_mask)
        cv2.imshow("world", world_img)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
