"""
check_camera_position.py  -  korer pa PC

Standalone tool to make sure the camera is mounted in the same position
and angle every time it's installed. Detects the 4 corners of the red
board border and compares them against a saved reference set of corners.
This is what stage 1 of SETUP_PLAN.md means by "must not move" - run this
any time the camera has been taken down and remounted, before trusting
any calibration done before the move.

Self-contained: it has its own small red-border + corner detector instead
of importing vision_bestfit (which doesn't exist on this branch yet).
It's deliberately simpler than a full vision pipeline - its only job is
"did the camera move", not precise robot navigation - so the red HSV
thresholds below are a starting point and may need adjusting for your
tape/lighting.

Usage:
  python check_camera_position.py --save
      Save the current detected corners as the reference. Do this once,
      right after mounting the camera in the position you want to keep
      using (ideally right after lens calibration).

  python check_camera_position.py
      Compare live corners against the saved reference. Green = saved
      reference, red = live detection. Nudge the camera until the red
      outline lands on the green one; per-corner pixel error is shown
      on screen.

Controls:
  SPACE  (in --save mode) capture & save the current corners
  ESC    quit
"""

import argparse
import json
import math
import os

import cv2
import numpy as np

CAMERA_INDEX = 0
REF_FILE     = "camera_position_ref.json"
TOL_PX       = 5   # corners within this many px of the reference = OK


def red_mask(hsv):
    """Two hue ranges since red wraps around 0/180 in OpenCV's HSV."""
    lower1 = np.array([0,   80, 60])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 80, 60])
    upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


def order_corners(pts):
    """Orders 4 points as TL/TR/BL/BR using x+y and x-y extremes."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return {
        "TL": (float(tl[0]), float(tl[1])),
        "TR": (float(tr[0]), float(tr[1])),
        "BL": (float(bl[0]), float(bl[1])),
        "BR": (float(br[0]), float(br[1])),
    }


def detect_corners(frame, mask):
    """Largest red contour -> 4-point polygon approximation -> ordered corners."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1000:
        return None

    peri = cv2.arcLength(cnt, True)
    for eps_frac in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1):
        approx = cv2.approxPolyDP(cnt, eps_frac * peri, True)
        if len(approx) == 4:
            return order_corners(approx.reshape(-1, 2))

    # Fallback if the border isn't approximating to a clean quad: use the
    # minimum-area rotated rectangle around the whole contour instead.
    rect = cv2.minAreaRect(cnt)
    box  = cv2.boxPoints(rect)
    return order_corners(box)


def draw_quad(img, corners, color, label):
    pts = np.array([corners["TL"], corners["TR"], corners["BR"], corners["BL"]],
                   dtype=np.int32)
    cv2.polylines(img, [pts], True, color, 2)
    for name, (x, y) in corners.items():
        p = (int(x), int(y))
        cv2.drawMarker(img, p, color, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(img, "{} {}".format(label, name), (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def open_camera(index):
    """Tries a few backends - DSHOW alone fails on some Windows setups
    with 'backend is generally available but can't be used to capture
    by index' even when the camera is fine."""
    for backend, name in [(cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DSHOW"),
                          (cv2.CAP_ANY, "default")]:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print("Opened camera {} via {}".format(index, name))
            return cap
        cap.release()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                         help="save the current corners as the reference")
    args = parser.parse_args()

    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        raise SystemExit(
            "Cannot open camera {}. Close any other app that might be "
            "using it (Windows Settings > Cameras preview, Teams, Logi "
            "Tune, browser tabs, etc.), check Settings > Privacy & "
            "security > Camera > 'Let desktop apps access your camera' "
            "is on, then try again.".format(CAMERA_INDEX))

    ref = None
    if not args.save:
        if not os.path.exists(REF_FILE):
            raise SystemExit(
                "No reference saved yet - run with --save first, once the "
                "camera is mounted where you want to keep using it.")
        with open(REF_FILE) as f:
            ref = json.load(f)
        print("Loaded reference from {}".format(REF_FILE))

    print("ESC to quit" + ("  |  SPACE to save reference" if args.save else ""))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = red_mask(hsv)
        corners = detect_corners(frame, mask)
        display = frame.copy()
        cv2.imshow("red mask (white = detected as red)", mask)

        if corners is None:
            cv2.putText(display, "Red border not detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        elif args.save:
            draw_quad(display, corners, (0, 255, 0), "live")
            cv2.putText(display, "SPACE = save as reference", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        else:
            draw_quad(display, ref, (0, 255, 0), "ref")
            draw_quad(display, corners, (0, 0, 255), "live")
            max_err = 0.0
            y = 30
            for name in ["TL", "TR", "BL", "BR"]:
                rx, ry = ref[name]
                lx, ly = corners[name]
                err = math.hypot(lx - rx, ly - ry)
                max_err = max(max_err, err)
                cv2.putText(display, "{}: {:.1f}px".format(name, err),
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if err <= TOL_PX else (0, 0, 255), 2)
                y += 25
            ok = max_err <= TOL_PX
            cv2.putText(display, "OK - same position" if ok else "MOVED - adjust camera",
                        (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if ok else (0, 0, 255), 2)

        cv2.imshow("camera position check", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if args.save and key == ord(' ') and corners is not None:
            with open(REF_FILE, "w") as f:
                json.dump(corners, f, indent=2)
            print("Saved reference corners to {}".format(REF_FILE))
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
