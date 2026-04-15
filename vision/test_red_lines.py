import cv2
from red_mask import red_mask
from red_lines import find_red_lines

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

colors = {
    "top": (255, 0, 0),
    "bottom": (0, 0, 255),
    "left": (0, 255, 0),
    "right": (0, 255, 255)
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)

    lines = find_red_lines(mask, frame.shape)

    if lines is not None:
        for name, line in lines.items():
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(frame, (x1, y1), (x2, y2), colors[name], 4)
                cv2.putText(
                    frame, name, (x1 + 10, y1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[name], 2
                )

    cv2.imshow("camera", frame)
    cv2.imshow("red mask", mask)



    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()