"""
TEST: Perspektiv-korrektion af robotposition
Korer: python tests/detection/test_perspective_correct.py

Viser BADE den ukorigerede (rod) og korrigerede (groen) robotposition
i world-view, saa du kan se hvor meget forskellen er.

Juster MARKER_HEIGHT_MM og HEADING_OFFSET oeverst i filen.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_bestfit"))

import cv2
import cv2.aruco as aruco
import numpy as np
import math

from red_mask import red_mask
from boundary_points import extract_boundary_points
from best_fit_lines import fit_frame_lines
from geometry import intersect_horizontal_vertical
from homography import (compute_homographies, pixel_to_world,
                        world_to_view, draw_world_grid,
                        BOARD_WIDTH_MM, BOARD_HEIGHT_MM)
from perspective_correct import (get_corrected_robot_pose,
                                  make_aruco_obj_pts)

# ── Konfiguration ─────────────────────────────────────────────────────────────
CAMERA_INDEX      = 1
MARKER_ID         = 0
MARKER_SIZE_MM    = 80
HEADING_OFFSET    = -90.0   # juster hvis pilen peger forkert
DISPLAY_SCALE     = 0.5
SMOOTH_ALPHA      = 0.85
# ─────────────────────────────────────────────────────────────────────────────

_dict     = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
_params   = aruco.DetectorParameters()
_detector = aruco.ArucoDetector(_dict, _params)
_obj_aruco = make_aruco_obj_pts(MARKER_SIZE_MM)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_corners = None

print("TEST: Perspektiv-korrektion")
print("Rod prik  = ukorigeret (ArUco pixel-center via homografi)")
print("Groen prik = korrigeret (strale til gulvplan)")
print("Forskel vises som linje imellem dem")
print("Q = afslut")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display   = frame.copy()
    world_img = np.zeros((int(BOARD_HEIGHT_MM*DISPLAY_SCALE),
                          int(BOARD_WIDTH_MM*DISPLAY_SCALE), 3), dtype=np.uint8)

    # ── Bane-hjorner ──────────────────────────────────────────────────────────
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)
    bp   = extract_boundary_points(mask)
    mdl  = fit_frame_lines(bp)

    corners       = None
    H_px_to_world = None
    H_px_to_view  = None

    if all(mdl[k] is not None for k in ["top","bottom","left","right"]):
        nc = {
            "TL": intersect_horizontal_vertical(mdl["top"],    mdl["left"]),
            "TR": intersect_horizontal_vertical(mdl["top"],    mdl["right"]),
            "BL": intersect_horizontal_vertical(mdl["bottom"], mdl["left"]),
            "BR": intersect_horizontal_vertical(mdl["bottom"], mdl["right"]),
        }
        if all(v is not None for v in nc.values()):
            if prev_corners is None:
                corners = nc
            else:
                corners = {n: (
                    int(SMOOTH_ALPHA*prev_corners[n][0]+(1-SMOOTH_ALPHA)*nc[n][0]),
                    int(SMOOTH_ALPHA*prev_corners[n][1]+(1-SMOOTH_ALPHA)*nc[n][1]),
                ) for n in nc}
            prev_corners = corners

            H_px_to_world, H_px_to_view, dw, dh = compute_homographies(
                corners, display_scale=DISPLAY_SCALE)
            world_raw = cv2.warpPerspective(frame, H_px_to_view, (dw, dh))
            world_img = world_raw.copy()
            draw_world_grid(world_img, display_scale=DISPLAY_SCALE, step_mm=200)

            # Tegn hjorner
            for name, pt in corners.items():
                cv2.circle(display, pt, 6, (0,255,0), -1)

    # ── Kameramatrix ──────────────────────────────────────────────────────────
    H_img, W_img = frame.shape[:2]
    f   = max(W_img, H_img)
    cam = np.array([[f,0,W_img/2],[0,f,H_img/2],[0,0,1]], dtype=np.float64)

    # ── Ukorigeret position (gammel metode: bare homografi pa pixel-center) ──
    raw_pos_mm   = None
    raw_center_px = None
    if H_px_to_world is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        c, ids, _ = _detector.detectMarkers(gray)
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid != MARKER_ID: continue
                mc = c[i][0]
                raw_center_px = (int(mc[:,0].mean()), int(mc[:,1].mean()))
                raw_pos_mm    = pixel_to_world(raw_center_px, H_px_to_world)

    # ── Korrigeret position (ny metode: strale til gulvplan) ─────────────────
    corrected = {"found": False}
    if corners is not None:
        corrected = get_corrected_robot_pose(
            frame, _detector, _obj_aruco,
            cam, corners,
            BOARD_WIDTH_MM, BOARD_HEIGHT_MM,
            heading_offset=HEADING_OFFSET,
            marker_id=MARKER_ID,
        )

    # ── Tegn i camera-view ────────────────────────────────────────────────────
    if raw_center_px:
        cv2.circle(display, raw_center_px, 8, (0,0,255), -1)
        cv2.putText(display, "Ukorigeret",
            (raw_center_px[0]+10, raw_center_px[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # ── Tegn i world-view ─────────────────────────────────────────────────────
    diff_mm = None

    if raw_pos_mm and H_px_to_world is not None:
        pv_raw = world_to_view(raw_pos_mm, DISPLAY_SCALE)
        cv2.circle(world_img, pv_raw, 8, (0,0,255), -1)
        cv2.putText(world_img, "RAW ({:.0f},{:.0f})".format(*raw_pos_mm),
            (pv_raw[0]+8, pv_raw[1]-8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,255), 1)

    if corrected["found"]:
        gx, gy    = corrected["pos_mm"]
        heading   = corrected["heading"]
        pv_corr   = world_to_view((gx, gy), DISPLAY_SCALE)

        # Tegn korrigeret position (groen)
        cv2.circle(world_img, pv_corr, 10, (0,255,0), -1)
        cv2.putText(world_img, "KORR ({:.0f},{:.0f})".format(gx, gy),
            (pv_corr[0]+8, pv_corr[1]+15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,0), 1)

        # Heading-pil i world-view
        ex = int(pv_corr[0] + 40*math.cos(math.radians(heading)))
        ey = int(pv_corr[1] - 40*math.sin(math.radians(heading)))
        cv2.arrowedLine(world_img, pv_corr, (ex,ey), (0,255,0), 2, tipLength=0.3)

        # Linje mellem raa og korrigeret
        if raw_pos_mm:
            pv_raw = world_to_view(raw_pos_mm, DISPLAY_SCALE)
            cv2.line(world_img, pv_raw, pv_corr, (255,255,0), 1)
            diff_mm = math.hypot(gx - raw_pos_mm[0], gy - raw_pos_mm[1])

        # Info i camera-view
        cv2.putText(display,
            "Korr: ({:.0f},{:.0f}) mm  heading: {:.1f} deg".format(gx, gy, heading),
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(display,
            "Kamerahoejde: {:.0f} mm".format(corrected.get("cam_height_mm", 0)),
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,0), 1)

    # ── Forskel og status ─────────────────────────────────────────────────────
    status_lines = []
    if corners is None:
        status_lines.append(("VENTER PA HJORNER", (0,100,255)))
    if not corrected["found"]:
        reason = corrected.get("reason","markoren ikke fundet")
        status_lines.append((reason, (0,0,255)))
    if diff_mm is not None:
        col = (0,255,0) if diff_mm < 30 else (0,165,255) if diff_mm < 80 else (0,0,255)
        status_lines.append(("Parallax fejl: {:.0f} mm".format(diff_mm), col))
        cv2.putText(world_img,
            "Parallax: {:.0f} mm".format(diff_mm),
            (10, world_img.shape[0]-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

    y = display.shape[0] - 20
    for txt, col in reversed(status_lines):
        cv2.putText(display, txt, (10,y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        y -= 28

    cv2.putText(world_img, "ROD=ukorigeret  GROEN=korrigeret  GUL=forskel",
        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

    cv2.imshow("Camera", display)
    cv2.imshow("World view", world_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()