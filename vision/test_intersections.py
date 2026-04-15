import cv2
import numpy as np

from red_mask import red_mask
from red_lines import find_red_lines

CAMERA_INDEX = 0

# --------- line intersection ---------
def intersect(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    A = np.array([
        [x2 - x1, x3 - x4],
        [y2 - y1, y3 - y4]
    ])
    B = np.array([
        x3 - x1,
        y3 - y1
    ])

    t, _ = np.linalg.lstsq(A, B, rcond=None)[0]

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return int(x), int(y)


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)

    lines = find_red_lines(mask, frame.shape)

    if lines is not None and None not in lines.values():

        top = lines["top"]
        bottom = lines["bottom"]
        left = lines["left"]
        right = lines["right"]

        # ---- draw lines ----
        cv2.line(frame, top[:2], top[2:], (255, 0, 0), 3)       # blue
        cv2.line(frame, bottom[:2], bottom[2:], (0, 0, 255), 3) # red
        cv2.line(frame, left[:2], left[2:], (0, 255, 0), 3)     # green
        cv2.line(frame, right[:2], right[2:], (0, 255, 255), 3) # yellow

        # ---- intersections ----
        top_left     = intersect(top, left)
        top_right    = intersect(top, right)
        bottom_left  = intersect(bottom, left)
        bottom_right = intersect(bottom, right)

        corners = {
            "TL": top_left,
            "TR": top_right,
            "BL": bottom_left,
            "BR": bottom_right
        }

        for name, (x, y) in corners.items():
            cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
            cv2.putText(
                frame, name, (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2
            )

    cv2.imshow("camera", frame)
    cv2.imshow("red mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()