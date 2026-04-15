import cv2
import numpy as np

from red_mask import red_mask
from line_fit import find_fitted_lines

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)

    fitted = find_fitted_lines(mask)

    for name, line in fitted.items():
        if line is None:
            continue

        vx, vy, x0, y0 = line

        # draw line
        x1 = int(x0 - 2000*vx)
        y1 = int(y0 - 2000*vy)
        x2 = int(x0 + 2000*vx)
        y2 = int(y0 + 2000*vy)

        cv2.line(frame, (x1,y1), (x2,y2), (0,255,255), 3)
        cv2.putText(frame, name, (int(x0), int(y0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("camera", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()