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
  [ / ]  decrease / increase manual focus (autofocus is disabled below -
         hunting between captures changes the effective focal length and
         wrecks the calibration, so focus is locked and set manually instead)
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
SQUARE_SIZE_MM = 23.0       # measure one square's side with a ruler
CAMERA_INDEX   = 0
FRAME_W        = 1280
FRAME_H        = 720
MIN_CAPTURES   = 15
OUTPUT_FILE    = "camera_calibration.json"

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

obj_points = []
img_points = []

def open_camera(index):
    """DSHOW is what reliably grabs frames at FRAME_W x FRAME_H for this
    camera - MSMF can open the handle but then fail every grabFrame call
    at this resolution, so it's listed only as a fallback, not preferred."""
    for backend, name in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"),
                          (cv2.CAP_ANY, "default")]:
        c = cv2.VideoCapture(index, backend)
        if c.isOpened():
            print("Opened camera {} via {}".format(index, name))
            return c
        c.release()
    return None


cap = open_camera(CAMERA_INDEX)
if cap is None:
    raise SystemExit("Cannot open camera {}".format(CAMERA_INDEX))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

# Some backends ignore property writes until the stream has actually
# started - read a few warmup frames before touching autofocus/focus.
warmup_ok = False
for _ in range(5):
    ret, _ = cap.read()
    warmup_ok = warmup_ok or ret
if not warmup_ok:
    cap.release()
    raise SystemExit(
        "Camera opened but can't grab frames at {}x{}. Try a lower "
        "resolution (edit FRAME_W/FRAME_H) or close other apps using the "
        "camera.".format(FRAME_W, FRAME_H))

# Autofocus hunting between captures changes the effective focal length,
# which calibrateCamera can't model (it assumes ONE fixed focal length for
# every image). Best-effort code-level disable - on this camera/driver it
# gets silently ignored via OpenCV, so the real fix is doing it at the
# OS/driver level BEFORE running this script: open Logi Tune (or the free
# "Webcam Settings" app from the Microsoft Store) and switch focus to
# manual there. [ / ] below still lets you nudge CAP_PROP_FOCUS in case it
# does take effect on your setup - harmless no-op if it doesn't.
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

# Pick up a focus value from autofocus_sweep.py if you've run it - saves
# re-finding a sharp value by hand every time this script starts.
try:
    with open("camera_focus.json") as f:
        focus_val = json.load(f)["focus"]
    cap.set(cv2.CAP_PROP_FOCUS, focus_val)
    print("Loaded focus={} from camera_focus.json (run autofocus_sweep.py "
          "again if this distance changes)".format(focus_val))
except FileNotFoundError:
    focus_val = cap.get(cv2.CAP_PROP_FOCUS)

if not cap.get(cv2.CAP_PROP_AUTOFOCUS) == 0:
    print("WARNING: camera/driver ignored the autofocus-off request via "
          "OpenCV - disable it in Logi Tune / Webcam Settings instead "
          "before capturing, or the calibration will be unreliable.")
FOCUS_STEP = 5

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
        print("Lost the camera feed (grabFrame failed) - stopping.")
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
    cv2.putText(display, "Focus: {:.0f}  ([ / ] to adjust)".format(focus_val),
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.imshow("calibrate", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit("Cancelled - no file saved.")

    elif key == ord('['):
        focus_val = max(0, focus_val - FOCUS_STEP)
        cap.set(cv2.CAP_PROP_FOCUS, focus_val)

    elif key == ord(']'):
        focus_val = min(255, focus_val + FOCUS_STEP)
        cap.set(cv2.CAP_PROP_FOCUS, focus_val)

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

if len(obj_points) == 0:
    raise SystemExit("No captures - nothing to calibrate.")

print("Calibrating with {} images...".format(len(obj_points)))
rms, cam_mat, dist_coef, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None)

print("RMS reprojection error: {:.4f} px (good: <0.5, ok: <1.0, bad: redo)".format(rms))
print("Camera matrix:\n{}".format(cam_mat))
print("Distortion coefficients:\n{}".format(dist_coef.ravel()))

# Overall RMS hides whether the error is spread evenly across every capture
# (a global problem - e.g. wrong square size, soft focus, non-flat board) or
# concentrated in a few bad ones (e.g. one capture flipped/rotated 180 deg,
# so the detector returned its corners in reverse order against objp - or
# motion blur on that specific shot). Per-view error tells you which.
print("\nPer-capture error (sorted worst first - look for outliers):")
per_view = []
for i in range(len(obj_points)):
    proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], cam_mat, dist_coef)
    err = cv2.norm(img_points[i], proj, cv2.NORM_L2) / (len(proj) ** 0.5)
    per_view.append((err, i + 1))
for err, idx in sorted(per_view, reverse=True):
    print("  capture #{}: {:.2f}px".format(idx, err))

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "frame_w":       FRAME_W,
        "frame_h":       FRAME_H,
        "camera_matrix": cam_mat.tolist(),
        "dist_coef":     dist_coef.ravel().tolist(),
        "rms_error":     rms,
    }, f, indent=2)

print("Saved to {}".format(OUTPUT_FILE))
