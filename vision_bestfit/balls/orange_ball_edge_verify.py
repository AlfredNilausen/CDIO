# orange_ball_edge_verify.py
import cv2
import numpy as np

from orange_intersection import compute_orange_intersection
from ball_edges import compute_ball_edges


EDGE_RATIO_THRESHOLD = 0.25
ANGLE_STEP = 6

def detect_orange_balls(frame):
    orange_intersection, _, _ = compute_orange_intersection(frame)
    edges = compute_ball_edges(frame)

    contours, _ = cv2.findContours(
        orange_intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    balls = []

    for cnt in contours:
        ok, info = is_orange_ball(cnt, edges)
        if not ok:
            continue

        cx, cy, r, ratio = info
        balls.append({
            "color": "orange",
            "x": cx,
            "y": cy,
            "r": r,
        })

    return balls


def circle_points(cx, cy, r, shape):
    pts = []
    for ang in range(0, 360, ANGLE_STEP):
        x = int(cx + r * np.cos(np.deg2rad(ang)))
        y = int(cy + r * np.sin(np.deg2rad(ang)))

        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            pts.append((x, y))
    return pts


def is_orange_ball(contour, ball_edges):
    (cx, cy), r = cv2.minEnclosingCircle(contour)
    cx, cy, r = int(cx), int(cy), int(r)

    if r < 5 or r > 20:
        return False, None

    pts = circle_points(cx, cy, r, ball_edges.shape)
    if not pts:
        return False, None

    hits = sum(1 for x, y in pts if ball_edges[y, x] > 0)
    ratio = hits / len(pts)

    return ratio >= EDGE_RATIO_THRESHOLD, (cx, cy, r, ratio)


# -------------------------------------------------
# TEST
# -------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("verified_orange_balls", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orange_intersection, _, _ = compute_orange_intersection(frame)
        edges = compute_ball_edges(frame)

        contours, _ = cv2.findContours(
            orange_intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        vis = frame.copy()

        for cnt in contours:
            ok, info = is_orange_ball(cnt, edges)
            if not ok:
                continue

            cx, cy, r, ratio = info
            cv2.circle(vis, (cx, cy), r, (0, 140, 255), 2)  # orange
            cv2.putText(
                vis, f"{int(ratio*100)}%",
                (cx - 15, cy - r - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 140, 255), 1
            )

        cv2.imshow("verified_orange_balls", vis)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
