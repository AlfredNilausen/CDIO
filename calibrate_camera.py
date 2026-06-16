"""
calibrate_camera.py  -  korer pa PC

One-time camera calibration using a printed checkerboard. Replaces the
guessed focal length in main.py with real, measured intrinsics.

Setup:
  1. Print a checkerboard pattern (any size) and tape it to something flat.
  2. Measure ONE square's side length in mm with a ruler, set SQUARE_SIZE_MM.
  3. Count the INNER corners (where 4 squares meet, not the squares
     themselves) along each side and set CHECKERBOARD = (cols, rows).
  4. Do NOT move the camera mount after this - the calibration is only
     valid for this exact camera position/lens/resolution.

Run:
  python calibrate_camera.py

Controls:
  SPACE  capture the current frame (only works once the board is detected)
  c      calibrate from all captures so far and save the result
  ESC    quit without saving

Move the checkerboard to a different position/angle/distance/tilt before
each capture - varied views are what make the calibration accurate.
Aim for 15-20 captures covering the whole frame (corners and edges too,
not just the center) and a range of tilts.
"""

import cv2
import numpy as np
import json

CHECKERBOARD   = (9, 6)     # inner corners (cols, rows) - count corners, not squares
SQUARE_SIZE_MM = 25.0       # measure one square's side with a ruler
CAMERA_INDEX   = 0
FRAME_W        = 1280
FRAME_H        = 720
MIN_CAPTURES   = 15
OUTPUT_FILE    = "camera_calibration.json"

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

obj_points = []
img_points = []

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
if not cap.isOpened():
    raise SystemExit("Cannot open camera {}".format(CAMERA_INDEX))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
find_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
              cv2.CALIB_CB_NORMALIZE_IMAGE |
              cv2.CALIB_CB_FAST_CHECK)

print("Keys: SPACE=capture  c=calibrate & save  ESC=quit")
print("Move the board between captures - vary position, angle and distance.")

gray = None
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, find_flags)

    display = frame.copy()
    if found:
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)
    cv2.putText(display, "{} captures - board {}".format(
        len(obj_points), "FOUND" if found else "not found"),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 0) if found else (0, 0, 255), 2)
    cv2.imshow("calibrate", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit("Cancelled - no file saved.")

    elif key == ord(' '):
        if found:
            corners_sub = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp.copy())
            img_points.append(corners_sub)
            print("Captured #{}".format(len(obj_points)))
        else:
            print("Board not found - move it fully into view and try again")

    elif key == ord('c'):
        if len(obj_points) < MIN_CAPTURES:
            print("Need at least {} captures, have {} - keep going".format(
                MIN_CAPTURES, len(obj_points)))
        else:
            break

cap.release()
cv2.destroyAllWindows()

print("Calibrating with {} images...".format(len(obj_points)))
rms, cam_mat, dist_coef, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None)

print("RMS reprojection error: {:.4f} px (good: <0.5, ok: <1.0, bad: redo)".format(rms))
print("Camera matrix:\n{}".format(cam_mat))
print("Distortion coefficients:\n{}".format(dist_coef.ravel()))

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "frame_w":       FRAME_W,
        "frame_h":       FRAME_H,
        "camera_matrix": cam_mat.tolist(),
        "dist_coef":     dist_coef.ravel().tolist(),
        "rms_error":     rms,
    }, f, indent=2)

print("Saved to {}".format(OUTPUT_FILE))
