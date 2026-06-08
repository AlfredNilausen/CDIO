# orange_candidates.py
import cv2
import numpy as np

from ball_edges import compute_ball_edges


# -----------------------------
# HOUGH (kandidater)
# -----------------------------
DP = 1.2
MIN_DIST = 18
CANNY_HIGH = 140
ACC_THRESH = 12

MIN_R = 7
MAX_R = 9

# -----------------------------
# EDGE-RING SCORE
# -----------------------------
RING_IN = 0.70
RING_OUT = 1.25
EDGE_DENS_MIN = 0.015
EDGE_PIX_MIN = 10

# -----------------------------
# RADIUS PRIOR
# -----------------------------
R_TARGET = 7
R_TOL = 3

TOP_N = 20
NMS_DIST = 14


def preprocess_for_hough(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bg = cv2.GaussianBlur(gray, (0, 0), 18)
    hp = cv2.addWeighted(gray, 1.8, bg, -0.8, 0)

    hp = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hp = cv2.GaussianBlur(hp, (7, 7), 1.2)

    return hp


def ring_mask(shape, cx, cy, r, inner_f, outer_f):
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    r1 = max(1, int(r * inner_f))
    r2 = max(r1 + 1, int(r * outer_f))
    cv2.circle(m, (cx, cy), r2, 255, -1)
    cv2.circle(m, (cx, cy), r1, 0, -1)
    return m


def edge_ring_score(edges, cx, cy, r):
    rm = ring_mask(edges.shape, cx, cy, r, RING_IN, RING_OUT)
    ring_pixels = int(np.count_nonzero(rm))
    if ring_pixels == 0:
        return 0.0, 0, 0

    edge_pixels = int(np.count_nonzero(cv2.bitwise_and(edges, edges, mask=rm)))
    dens = edge_pixels / ring_pixels
    return dens, edge_pixels, ring_pixels


def radius_prior(r):
    d = abs(r - R_TARGET)
    if d > R_TOL:
        return 0.0
    return 1.0 - d / (R_TOL + 1e-6)


def nms(cands, dist=NMS_DIST):
    kept = []
    for c in cands:
        x, y, r, score, meta = c
        ok = True
        for k in kept:
            kx, ky, _, _, _ = k
            if (x - kx) ** 2 + (y - ky) ** 2 < dist ** 2:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


def compute_orange_candidates(frame):
    """
    Returnerer cirkel-kandidater som liste af dicts:
    [{"x":..,"y":..,"r":..,"score":..,"dens":..,"epx":..,"rp":..}, ...]
    """
    pre = preprocess_for_hough(frame)
    edges = compute_ball_edges(frame)

    circles = cv2.HoughCircles(
        pre,
        cv2.HOUGH_GRADIENT,
        dp=DP,
        minDist=MIN_DIST,
        param1=CANNY_HIGH,
        param2=ACC_THRESH,
        minRadius=MIN_R,
        maxRadius=MAX_R
    )

    scored = []

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))

        for (x, y, r) in circles:
            x, y, r = int(x), int(y), int(r)

            dens, epx, _ = edge_ring_score(edges, x, y, r)
            if epx < EDGE_PIX_MIN:
                continue
            if dens < EDGE_DENS_MIN:
                continue

            rp = radius_prior(r)
            score = 0.65 * min(1.0, dens / (EDGE_DENS_MIN + 1e-6)) + 0.35 * rp

            scored.append((x, y, r, float(score), (dens, epx, rp)))

    scored.sort(key=lambda t: t[3], reverse=True)
    scored = nms(scored, dist=NMS_DIST)
    best = scored[:TOP_N]

    out = []
    for (x, y, r, score, meta) in best:
        dens, epx, rp = meta
        out.append({
            "x": x,
            "y": y,
            "r": r,
            "score": score,
            "dens": dens,
            "epx": epx,
            "rp": rp
        })

    return out


# -------------------------------------------------
# TEST / VIS
# -------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("orange_candidates", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        edges = compute_ball_edges(frame)
        cands = compute_orange_candidates(frame)

        vis = frame.copy()

        for c in cands:
            x, y, r = c["x"], c["y"], c["r"]
            score, dens, epx = c["score"], c["dens"], c["epx"]
            cv2.circle(vis, (x, y), r, (0, 165, 255), 2)
            cv2.putText(
                vis,
                f"r={r} sc={score:.2f} d={dens:.3f} e={epx}",
                (x - 70, y - r - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 165, 255),
                1
            )

        cv2.imshow("orange_candidates", vis)
        cv2.imshow("edges", edges)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
