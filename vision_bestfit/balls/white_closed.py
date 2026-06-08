# white_closed.py
import numpy as np
import cv2

from ball_edges import compute_ball_edges


def compute_white_closed(frame):
    """
    Input:  BGR frame
    Output: white_closed (uint8) – alle lukkede former fyldt
    """

    edges = compute_ball_edges(frame)

    # === VIGTIGT: luk små huller i edge-figurer ===
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find alle eksterne konturer
    contours, _ = cv2.findContours(
        edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    white_closed = np.zeros(edges.shape, dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Kun areal-filtrering (ingen formforståelse!)
        if area < 80:
            continue

        cv2.drawContours(
            white_closed, [cnt], -1, 255, thickness=cv2.FILLED
        )

    return white_closed, edges, edges_closed


# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edges_closed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("white_closed", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        white_closed, edges, edges_closed = compute_white_closed(frame)

        cv2.imshow("camera", frame)
        cv2.imshow("edges", edges)
        cv2.imshow("edges_closed", edges_closed)
        cv2.imshow("white_closed", white_closed)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
