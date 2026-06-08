import cv2
import numpy as np
import math
from collections import deque

# -------------------------
# TUNING / STARTVÆRDIER
# -------------------------
CROP_BORDER = 10           # crop lidt kant (rammen ligger ofte ved kanten)
CLAHE_CLIP = 2.2
CLAHE_GRID = (8, 8)

# Canny (sænk hvis hvid bold er svag)
CANNY1 = 45
CANNY2 = 130

# Line-removal (rette kanter)
LINE_LEN = 40              # min længde (px) for at regne som "ret kant"
LINE_THICK = 5             # tykkelse af maskelinje (2..6)
LINE_DILATE = 2            # udvid linjemaske (0..2)

USE_LSD = True             # prøv LSD (kræver ofte opencv-contrib)
USE_MORPH_OPEN_TOO = True  # kombiner LSD/Hough med morfologi (god til rammer)

# Edge connect (samle brudte boldkanter)
DILATE_ITERS = 1
CLOSE_ITERS = 2

# Kandidatfiltre (radius læres automatisk)
R_MIN_INIT = 4
R_MAX_INIT = 80

# Edge-support: hvor meget ring-kant der skal være edge
RING_THICK = 2
MIN_EDGE_SUPPORT = 0.08    # sænk hvis boldkant er meget brudt (0.06..0.15)

# Auto-radius lock (median ±2 px)
LOCK_AFTER = 8
LOCK_TOL = 2.0
R_HIST_LEN = 30

# -------------------------
# RANSAC circle fit params
# -------------------------
RANSAC_ITERS = 700
RANSAC_DIST_THR = 2.8        # px tolerence til cirkelring (2..4 typisk)
MIN_INLIER_RATIO = 0.70      # dit krav
MIN_ARC_DEG = 170            # kræv vinkel-dækning (120..220)
MIN_POINTS_COMP = 70         # min edge-punkter pr component
MAX_POINTS_PER_COMP = 2500   # subsample for fart

# -------------------------
# Kernels
# -------------------------
K_DIL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K_H = cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_LEN, 1))
K_V = cv2.getStructuringElement(cv2.MORPH_RECT, (1, LINE_LEN))


# -------------------------
# Hjælpefunktioner
# -------------------------
def preprocess(gray):
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    g = clahe.apply(gray)
    g = cv2.medianBlur(g, 5)
    return g

def ring_mask(shape_hw, x, y, r, thick=2):
    h, w = shape_hw
    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    r_out = r + thick
    r_in = max(1, r - thick)
    ring = (dist2 <= r_out * r_out) & (dist2 >= r_in * r_in)
    return (ring.astype(np.uint8) * 255)

def edge_support_score(edges, x, y, r, thick=2):
    h, w = edges.shape[:2]
    y0, y1 = max(0, y - r - thick), min(h, y + r + thick + 1)
    x0, x1 = max(0, x - r - thick), min(w, x + r + thick + 1)
    roi = edges[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
    cx, cy = x - x0, y - y0
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

    r_out = r + thick
    r_in = max(1, r - thick)
    ring = (dist2 <= r_out * r_out) & (dist2 >= r_in * r_in)

    ring_count = int(ring.sum())
    if ring_count == 0:
        return 0.0

    edge_count = int((roi > 0)[ring].sum())
    return edge_count / ring_count

def classify_orange_white(frame_bgr, x, y, r):
    """Robust farveklassifikation inde i cirklen (median + ignore highlights)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

    h, w = hsv.shape[:2]
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)

    roi_hsv = hsv[y0:y1, x0:x1]
    roi_lab = lab[y0:y1, x0:x1]
    if roi_hsv.size == 0:
        return "BALL", 0.0

    yy, xx = np.ogrid[:roi_hsv.shape[0], :roi_hsv.shape[1]]
    cx, cy = x - x0, y - y0
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if circle.sum() < 30:
        return "BALL", 0.0

    H = roi_hsv[..., 0][circle]
    S = roi_hsv[..., 1][circle]
    V = roi_hsv[..., 2][circle]

    L = roi_lab[..., 0][circle]
    A = roi_lab[..., 1][circle]
    B = roi_lab[..., 2][circle]
    ab_dist = np.abs(A.astype(np.int16) - 128) + np.abs(B.astype(np.int16) - 128)

    # ignore de mest overeksponerede pixels (refleks)
    v_cut = np.percentile(V, 95)
    keep = V <= v_cut
    if keep.sum() > 30:
        H, S, V = H[keep], S[keep], V[keep]
        L, ab_dist = L[keep], ab_dist[keep]

    h_med = float(np.median(H))
    s_med = float(np.median(S))
    v_med = float(np.median(V))
    l_med = float(np.median(L))
    ab_med = float(np.median(ab_dist))

    # orange score
    orange_h = 1.0 if (5 <= h_med <= 30) else 0.0
    orange_s = np.clip((s_med - 50) / 120.0, 0, 1)
    orange_score = 0.6 * orange_h + 0.4 * orange_s

    # white score
    white_neutral = np.clip((60 - ab_med) / 60.0, 0, 1)
    white_light = np.clip((l_med - 120) / 80.0, 0, 1)
    white_low_sat = np.clip((80 - s_med) / 80.0, 0, 1)
    white_score = 0.45 * white_neutral + 0.35 * white_light + 0.20 * white_low_sat

    if orange_score >= white_score and orange_score > 0.35:
        return "ORANGE", float(orange_score)
    if white_score > 0.35:
        return "WHITE", float(white_score)
    return "BALL", float(max(orange_score, white_score))

# -------------------------
# Line removal: LSD (alle vinkler) + fallback til Hough
# -------------------------
def remove_long_lines_v2(edges, gray_for_lsd):
    """
    Fjern lange rette linjesegmenter (alle vinkler) fra edges.
    Returner (edges_clean, lines_mask)
    """
    h, w = edges.shape[:2]

    line_mask = np.zeros((h, w), dtype=np.uint8)

    # (A) LSD (hvis muligt)
    did_any = False
    if USE_LSD:
        try:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
            lines = lsd.detect(gray_for_lsd)[0]  # Nx1x4 eller None
            if lines is not None:
                for l in lines:
                    x0, y0, x1, y1 = l[0]
                    length = math.hypot(x1 - x0, y1 - y0)
                    if length >= LINE_LEN:
                        cv2.line(
                            line_mask,
                            (int(round(x0)), int(round(y0))),
                            (int(round(x1)), int(round(y1))),
                            255,
                            LINE_THICK
                        )
                did_any = True
        except Exception:
            did_any = False

    # (B) Fallback: HoughLinesP på edges (ofte OK)
    if not did_any:
        # Hough parametre kan tunes
        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                                 minLineLength=LINE_LEN, maxLineGap=6)
        if linesP is not None:
            for l in linesP:
                x0, y0, x1, y1 = l[0]
                cv2.line(line_mask, (x0, y0), (x1, y1), 255, LINE_THICK)

    # (C) Valgfrit: din gamle morfologi (open) til vandret/lodret rammer
    if USE_MORPH_OPEN_TOO:
        horiz = cv2.morphologyEx(edges, cv2.MORPH_OPEN, K_H, iterations=1)
        vert = cv2.morphologyEx(edges, cv2.MORPH_OPEN, K_V, iterations=1)
        line_mask = cv2.bitwise_or(line_mask, cv2.bitwise_or(horiz, vert))

    # (D) Udvid masken lidt så vi fjerner hele kant-tykkelsen
    if LINE_DILATE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        line_mask = cv2.dilate(line_mask, k, iterations=LINE_DILATE)

    edges_clean = cv2.bitwise_and(edges, cv2.bitwise_not(line_mask))
    return edges_clean, line_mask


# -------------------------
# RANSAC circle fit på edgepunkter
# -------------------------
def circle_from_3pts(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3
    e = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / 2.0
    f = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / 2.0

    det = a * d - b * c
    if abs(det) < 1e-8:
        return None

    cx = (d * e - b * f) / det
    cy = (-c * e + a * f) / det
    r = math.hypot(x1 - cx, y1 - cy)
    return cx, cy, r

def angular_coverage_deg(points_xy, cx, cy):
    ang = np.arctan2(points_xy[:, 1] - cy, points_xy[:, 0] - cx)
    ang = np.sort(ang)
    if ang.size < 2:
        return 0.0
    gaps = np.diff(ang)
    wrap_gap = (ang[0] + 2 * np.pi) - ang[-1]
    max_gap = max(float(np.max(gaps)), float(wrap_gap))
    coverage = 2 * np.pi - max_gap
    return float(np.degrees(coverage))

def ransac_circle(points_xy, radius_est=None):
    n = points_xy.shape[0]
    if n < 20:
        return None

    # subsample for speed
    if n > MAX_POINTS_PER_COMP:
        idx = np.random.choice(n, MAX_POINTS_PER_COMP, replace=False)
        pts = points_xy[idx]
    else:
        pts = points_xy

    m = pts.shape[0]
    best = None

    for _ in range(RANSAC_ITERS):
        i = np.random.choice(m, 3, replace=False)
        model = circle_from_3pts(pts[i[0]], pts[i[1]], pts[i[2]])
        if model is None:
            continue

        cx, cy, r = model

        if not (R_MIN_INIT <= r <= R_MAX_INIT):
            continue

        # radius lock (blød gating)
        if radius_est is not None:
            if abs(r - radius_est) > max(LOCK_TOL, 1.0) * 4.0:
                continue

        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        resid = np.abs(d - r)
        inliers = resid < RANSAC_DIST_THR
        inlier_count = int(np.sum(inliers))
        ratio = inlier_count / float(m)
        if ratio < MIN_INLIER_RATIO:
            continue

        inlier_pts = pts[inliers]
        arc = angular_coverage_deg(inlier_pts, cx, cy)
        if arc < MIN_ARC_DEG:
            continue

        rad_pen = 0.0
        if radius_est is not None:
            rad_pen = min(abs(r - radius_est) / max(LOCK_TOL, 1.0), 3.0)

        # score: ratio tungt + arc let + radius penalty
        score = 2.2 * ratio + 0.010 * arc - 0.25 * rad_pen

        if best is None or score > best["score"]:
            best = {
                "cx": float(cx),
                "cy": float(cy),
                "r": float(r),
                "ratio": float(ratio),
                "arc": float(arc),
                "score": float(score),
                "inliers": inlier_count,
                "total": int(m)
            }

    return best

def detect_circle_ransac_from_edges(edges_bin, radius_est=None):
    """
    Kør RANSAC pr connected component for at gøre '70% inliers' meningsfuldt.
    edges_bin: binary edge image (0/255)
    """
    bin_img = (edges_bin > 0).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    best = None

    for comp_id in range(1, num):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < MIN_POINTS_COMP:
            continue

        ys, xs = np.where(labels == comp_id)
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

        cand = ransac_circle(pts, radius_est=radius_est)
        if cand is None:
            continue

        if best is None or cand["score"] > best["score"]:
            best = cand

    return best


# -------------------------
# Detection pipeline (edges -> remove lines -> connect -> RANSAC -> score)
# -------------------------
def detect_circle(frame_bgr, radius_est=None):
    H, W = frame_bgr.shape[:2]
    cb = CROP_BORDER
    work = frame_bgr[cb:H - cb, cb:W - cb] if cb > 0 else frame_bgr

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    g = preprocess(gray)

    edges_raw = cv2.Canny(g, CANNY1, CANNY2)

    edges_clean, lines_mask = remove_long_lines_v2(edges_raw, g)

    # connect brudte kanter
    e = cv2.dilate(edges_clean, K_DIL, iterations=DILATE_ITERS)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, K_CLOSE, iterations=CLOSE_ITERS)

    # RANSAC på connected components i e (forbundet)
    cand = detect_circle_ransac_from_edges(e, radius_est=radius_est)

    best = None
    if cand is not None:
        xi = int(round(cand["cx"]))
        yi = int(round(cand["cy"]))
        ri = int(round(cand["r"]))

        # score edge-support på edges_raw (ikke på dilate/close)
        es = edge_support_score(edges_raw, xi, yi, ri, thick=RING_THICK)
        if es >= MIN_EDGE_SUPPORT:
            best = {
                "x": xi,
                "y": yi,
                "r": float(cand["r"]),
                "edge_support": float(es),
                "quality": float(cand["score"] + 1.6 * es),
                "ratio": float(cand["ratio"]),
                "arc": float(cand["arc"]),
                "cb": cb
            }

    # pak debug til fuld størrelse
    edges_full = np.zeros((H, W), dtype=np.uint8)
    clean_full = np.zeros((H, W), dtype=np.uint8)
    conn_full = np.zeros((H, W), dtype=np.uint8)
    lines_full = np.zeros((H, W), dtype=np.uint8)

    if cb > 0:
        edges_full[cb:H - cb, cb:W - cb] = edges_raw
        clean_full[cb:H - cb, cb:W - cb] = edges_clean
        conn_full[cb:H - cb, cb:W - cb] = e
        lines_full[cb:H - cb, cb:W - cb] = lines_mask
        if best is not None:
            best["x"] += cb
            best["y"] += cb
    else:
        edges_full = edges_raw
        clean_full = edges_clean
        conn_full = e
        lines_full = lines_mask

    return best, edges_full, clean_full, conn_full, lines_full


# -------------------------
# MAIN
# -------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Kunne ikke åbne kamera")

    r_hist = deque(maxlen=R_HIST_LEN)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        radius_est = None
        if len(r_hist) >= LOCK_AFTER:
            radius_est = float(np.median(np.array(r_hist)))

        best, edges_full, clean_full, conn_full, lines_full = detect_circle(frame, radius_est=radius_est)

        vis = frame.copy()
        circle_edges_best = np.zeros_like(edges_full)

        if best is not None:
            x, y = best["x"], best["y"]
            r = int(round(best["r"]))
            r_hist.append(best["r"])

            rm = ring_mask(edges_full.shape, x, y, r, thick=RING_THICK)
            circle_edges_best = cv2.bitwise_and(edges_full, rm)

            label, conf = classify_orange_white(frame, x, y, r)

            col = (0, 255, 0) if label == "ORANGE" else (255, 255, 255) if label == "WHITE" else (0, 255, 255)
            cv2.circle(vis, (x, y), r, col, 2)
            cv2.circle(vis, (x, y), 2, (0, 255, 255), -1)

            txt = (f"{label} r={best['r']:.1f}px "
                   f"edge={best['edge_support']:.2f} "
                   f"inl={best['ratio']:.2f} arc={best['arc']:.0f}deg conf={conf:.2f}")
            cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            if radius_est is not None:
                cv2.putText(vis, f"radius_est ~ {radius_est:.1f} +/- {LOCK_TOL}px",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(vis, "No circle candidate", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("result", vis)
        cv2.imshow("edges_raw", edges_full)
        cv2.imshow("edges_clean", clean_full)
        cv2.imshow("edges_connected", conn_full)
        cv2.imshow("lines_mask", lines_full)
        cv2.imshow("circle_edges_best", circle_edges_best)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()