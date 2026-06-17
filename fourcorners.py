#!/usr/bin/env python3

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

# relative to robot center (mm)
MARKER_OFFSETS_MM = {
    "FL": (0,  -85),
    "FR": ( 0,  85),
    "BL": (-370, -85),
    "BR": ( -370, 85),
}

# green detection
GREEN_H_MIN = 25
GREEN_H_MAX = 95
GREEN_S_MIN = 40
GREEN_V_MIN = 40

MIN_BLOB_AREA = 10

ROBOT_MARKER_ID = 0
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# ------------------------------------------------------------------
# IMPORT THESE FROM YOUR EXISTING PROJECT
# ------------------------------------------------------------------

from red_mask import red_mask
from boundary_points import extract_boundary_points
from best_fit_lines import fit_frame_lines

# ------------------------------------------------------------------
# ARUCO HEADING
# ------------------------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    aruco_params
)

MARKER_SIZE_MM = 80
_half = MARKER_SIZE_MM / 2.0

OBJ_PTS = np.array([
    [-_half,  _half, 0],
    [ _half,  _half, 0],
    [ _half, -_half, 0],
    [-_half, -_half, 0],
], dtype=np.float32)


def detect_heading(frame):

    H, W = frame.shape[:2]

    f = max(H, W)

    cam = np.array([
        [f, 0, W/2],
        [0, f, H/2],
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

        heading = math.degrees(
            math.atan2(
                -R[1, 0],
                R[0, 0]
            )
        ) 

        center = (
            int(mc[:,0].mean()),
            int(mc[:,1].mean())
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

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5,5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    cnts, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

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
# ROTATE OFFSET
# ------------------------------------------------------------------

def rotate_offset(offset_mm, heading_deg):

    x, y = offset_mm

    a = math.radians(heading_deg)

    xr = (
        x * math.cos(a)
        - y * math.sin(a)
    )

    yr = (
        x * math.sin(a)
        + y * math.cos(a)
    )

    return xr, yr


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

cap = cv2.VideoCapture(1)

while True:

    ok, frame = cap.read()

    if not ok:
        break

    display = frame.copy()

    # -------------------------------------------------------------
    # board detection
    # -------------------------------------------------------------

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_red = red_mask(hsv)

    bp = extract_boundary_points(mask_red)

    mdl = fit_frame_lines(bp)

    required = ["top","bottom","left","right"]

    if not all(mdl[k] is not None for k in required):

        cv2.imshow("camera", display)

        if cv2.waitKey(1) == 27:
            break

        continue

    corners = {
        "TL": intersect_horizontal_vertical(
            mdl["top"],
            mdl["left"]
        ),
        "TR": intersect_horizontal_vertical(
            mdl["top"],
            mdl["right"]
        ),
        "BL": intersect_horizontal_vertical(
            mdl["bottom"],
            mdl["left"]
        ),
        "BR": intersect_horizontal_vertical(
            mdl["bottom"],
            mdl["right"]
        ),
    }

    if any(v is None for v in corners.values()):
        continue

    H_px_to_world, H_px_to_view, disp_w, disp_h = \
        compute_homographies(
            corners,
            DISPLAY_SCALE
        )

    world_img = cv2.warpPerspective(
        frame,
        H_px_to_view,
        (disp_w, disp_h)
    )

    draw_world_grid(
        world_img,
        DISPLAY_SCALE
    )

    # -------------------------------------------------------------
    # heading
    # -------------------------------------------------------------

    aruco_pose = detect_heading(frame)

    if aruco_pose is None:

        cv2.imshow("camera", display)
        cv2.imshow("world", world_img)

        if cv2.waitKey(1) == 27:
            break

        continue

    heading = aruco_pose["heading"]

    # -------------------------------------------------------------
    # green blobs
    # -------------------------------------------------------------

    blobs_px, green_mask = detect_green_blobs(frame)

    for p in blobs_px:
        cv2.circle(display, p, 10, (0,255,0), 2)

    centers = []

    # -------------------------------------------------------------
    # ASSIGNMENT  (heading-aware, works for 2 or 4 markers)
    #
    # Uses ArUco heading to tell front blobs from back blobs.
    # Physical markers: FL (front-left) and BR (back-right).
    # With 4 blobs also FR and BL are used.
    # -------------------------------------------------------------

    if len(blobs_px) >= 2:

        # Convert ArUco center to world to get reference point for direction
        aruco_world = pixel_to_world(aruco_pose["center_px"], H_px_to_world)
        ax, ay = aruco_world
        hx = math.cos(math.radians(heading))
        hy = math.sin(math.radians(heading))

        # Score each blob by how far forward it is from the ArUco center.
        # This correctly identifies front vs back regardless of robot orientation.
        blobs_world = [pixel_to_world(p, H_px_to_world) for p in blobs_px]
        scored = sorted(
            blobs_world,
            key=lambda b: (b[0] - ax) * hx + (b[1] - ay) * hy,
            reverse=True  # most forward first
        )

        # Most forward blob = FL, most backward = BR (diagonal pair)
        for label, (bx, by) in [("FL", scored[0]), ("BR", scored[-1])]:
            off_x, off_y = rotate_offset(MARKER_OFFSETS_MM[label], heading)
            centers.append((bx - off_x, by - off_y))

        # If all 4 blobs visible, use FR (2nd forward) and BL (2nd backward) too
        if len(scored) >= 4:
            for label, (bx, by) in [("FR", scored[1]), ("BL", scored[-2])]:
                off_x, off_y = rotate_offset(MARKER_OFFSETS_MM[label], heading)
                centers.append((bx - off_x, by - off_y))

    if centers:

        rx = sum(p[0] for p in centers) / len(centers)
        ry = sum(p[1] for p in centers) / len(centers)

        robot_view = world_to_view(
            (rx, ry),
            DISPLAY_SCALE
        )

        cv2.circle(
            world_img,
            robot_view,
            8,
            (255,0,0),
            -1
        )

        arrow_len = 120

        ex = rx + arrow_len * math.cos(
            math.radians(heading)
        )

        ey = ry + arrow_len * math.sin(
            math.radians(heading)
        )

        cv2.arrowedLine(
            world_img,
            world_to_view((rx, ry)),
            world_to_view((ex, ey)),
            (0,0,255),
            3
        )

        cv2.putText(
            world_img,
            "X={:.0f} Y={:.0f} H={:.1f}".format(
                rx,
                ry,
                heading
            ),
            (20,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        print(
            "\rRobot: ({:.0f},{:.0f}) heading={:.1f}".format(
                rx,
                ry,
                heading
            ),
            end=""
        )

    cv2.imshow("camera", display)
    cv2.imshow("green mask", green_mask)
    cv2.imshow("world", world_img)

    k = cv2.waitKey(1)

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()