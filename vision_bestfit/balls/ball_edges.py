# ball_edges.py
import cv2
import numpy as np

def compute_ball_edges(frame):
    """
    Input: BGR frame
    Output: edges_clean (uint8)
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    v = clahe.apply(v)

    v = cv2.medianBlur(v, 7)
    v = cv2.GaussianBlur(v, (9, 9), 1.0)

    edges = cv2.Canny(v, 50, 150)

    # fjern lange linjer (banen)
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    edges = cv2.subtract(edges, cv2.morphologyEx(edges, cv2.MORPH_OPEN, k_h))
    edges = cv2.subtract(edges, cv2.morphologyEx(edges, cv2.MORPH_OPEN, k_v))

    # Let ekstra forstærkning for at lukke figurer
    edges_clean = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=2  # <-- 1 ekstra pixel
    )

    return edges_clean


# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("balledges", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        edges = compute_ball_edges(frame)

        cv2.imshow("camera", frame)
        cv2.imshow("balledges", edges)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
