import cv2
import numpy as np

from red_mask import red_mask
from line_fit import find_fitted_lines
from geometry import intersect

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)

    fitted = find_fitted_lines(mask)

    if None not in fitted.values():
        TL = intersect(fitted["top"], fitted["left"])
        TR = intersect(fitted["top"], fitted["right"])
        BL = intersect(fitted["bottom"], fitted["left"])
        BR = intersect(fitted["bottom"], fitted["right"])

        for name, p in zip(
            ["TL","TR","BL","BR"],
            [TL,TR,BL,BR]
        ):
            cv2.circle(frame, p, 10, (0,255,0), -1)
            cv2.putText(frame, name, (p[0]+8,p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("camera", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()