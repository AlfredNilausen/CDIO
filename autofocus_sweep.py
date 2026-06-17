"""
autofocus_sweep.py  -  korer pa PC

Finds a sharp manual focus value automatically, without dragging any
slider by hand. Sweeps CAP_PROP_FOCUS across its full range and scores
each value with the variance of the Laplacian (a standard no-reference
sharpness metric - more edge contrast = higher score = sharper image).
Locks the camera to whichever value scored highest.

Point the camera at the checkerboard (or anything with texture/edges,
e.g. the board itself) at the actual distance you'll calibrate/operate
from before running this - the result is only valid for that distance.

Run:
  python autofocus_sweep.py

Saves the winning value to camera_focus.json. calibrate_camera.py picks
this up automatically on startup if the file exists.
"""

import cv2
import json
import time

CAMERA_INDEX   = 0
FRAME_W        = 1280
FRAME_H        = 720
FOCUS_MIN      = 0
FOCUS_MAX      = 255
FOCUS_STEP     = 5
SETTLE_SEC     = 0.4   # time for the focus motor to physically stop moving
SCORE_FRAMES   = 3     # frames averaged per value, after settling
OUTPUT_FILE    = "camera_focus.json"


def open_camera(index):
    for backend, name in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"),
                          (cv2.CAP_ANY, "default")]:
        c = cv2.VideoCapture(index, backend)
        if c.isOpened():
            print("Opened camera {} via {}".format(index, name))
            return c
        c.release()
    return None


def sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        raise SystemExit("Cannot open camera {}".format(CAMERA_INDEX))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    print("Sweeping focus {}..{} (step {}) - keep the board still, this "
          "takes a while because each step waits for the lens motor to "
          "actually stop moving...".format(FOCUS_MIN, FOCUS_MAX, FOCUS_STEP))

    best_val, best_score = None, -1.0
    for val in range(FOCUS_MIN, FOCUS_MAX + 1, FOCUS_STEP):
        cap.set(cv2.CAP_PROP_FOCUS, val)
        time.sleep(SETTLE_SEC)
        cap.read()  # flush one possibly-stale buffered frame

        scores = []
        for _ in range(SCORE_FRAMES):
            ret, frame = cap.read()
            if ret:
                scores.append(sharpness(frame))
        if not scores:
            print("  focus={:3d}  (no frame - skipped)".format(val))
            continue
        score = sum(scores) / len(scores)
        print("  focus={:3d}  sharpness={:.1f}".format(val, score))
        if score > best_score:
            best_score = score
            best_val = val

    if best_val is None:
        cap.release()
        raise SystemExit("Could not read any frames - is the camera in use elsewhere?")

    cap.set(cv2.CAP_PROP_FOCUS, best_val)
    time.sleep(SETTLE_SEC)
    print("\nBest focus value: {} (sharpness {:.1f})".format(best_val, best_score))

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"focus": best_val}, f, indent=2)
    print("Saved to {}".format(OUTPUT_FILE))

    cap.read()
    ret, frame = cap.read()
    if ret:
        cv2.putText(frame, "Best focus: {} - press any key to close".format(best_val),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("autofocus_sweep result", frame)
        cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
