# white_mask.py
import cv2
import numpy as np

def compute_white_mask(frame):
    """
    Input: BGR frame
    Output: white_mask (uint8)
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Hvid = næsten ingen farve + høj lyshed
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([180, 50, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Fjern støj
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("white_mask", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        white_mask = compute_white_mask(frame)

        cv2.imshow("camera", frame)
        cv2.imshow("white_mask", white_mask)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
