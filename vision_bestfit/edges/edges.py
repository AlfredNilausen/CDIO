import numpy as np
import cv2
import math

# =====================================================
# KAMERA
# =====================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("edges_clean", cv2.WINDOW_NORMAL)
cv2.namedWindow("detected_balls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("detected_balls", 1280, 720)

# =====================================================
# EDGE / PREPROCESS
# =====================================================
CANNY_LOW = 50
CANNY_HIGH = 150
LINE_KERNEL_LEN = 25
THICKEN_KERNEL = (3, 3)

clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))

# =====================================================
# GEOMETRI
# =====================================================
RADIUS = 8
RING_THICK = 1
STEP = 2
COVER_REQ = 0.30
SKIP_RADIUS = 6

# =====================================================
# FARVE (ROBUST HSV)
# =====================================================
# ORANGE (klar + bleg)
ORANGE_H_MIN = 12
ORANGE_H_MAX = 26
ORANGE_S_STRONG = 110
ORANGE_S_WEAK = 50
ORANGE_V_MIN = 80

ORANGE_OVERRIDE_RATIO = 0.10  # 15 %


# HVID (mere tolerant)
WHITE_S_MAX = 110
WHITE_V_MIN = 130

COLOR_COVER_REQ = 0.60

# =====================================================
# HJÆLPEFUNKTIONER
# =====================================================
def make_ring_offsets(r, thick):
    offs = []
    rr = r + thick
    for dy in range(-rr, rr + 1):
        for dx in range(-rr, rr + 1):
            if abs(math.hypot(dx, dy) - r) <= thick:
                offs.append((dx, dy))
    return (
        np.array([o[0] for o in offs], np.int16),
        np.array([o[1] for o in offs], np.int16)
    )


def make_disk_offsets(r):
    offs = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                offs.append((dx, dy))
    return (
        np.array([o[0] for o in offs], np.int16),
        np.array([o[1] for o in offs], np.int16)
    )


ring_dx, ring_dy = make_ring_offsets(RADIUS, RING_THICK)
skip_dx, skip_dy = make_disk_offsets(SKIP_RADIUS)

# =====================================================
# MAIN LOOP
# =====================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # EDGE PIPELINE
    # -----------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_chan = hsv[:, :, 2]
    v_contrast = clahe.apply(v_chan)

    blur = cv2.GaussianBlur(
        cv2.medianBlur(v_contrast, 7),
        (9, 9),
        1.0
    )

    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_KERNEL_LEN, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, LINE_KERNEL_LEN))
    edges = cv2.subtract(edges, cv2.morphologyEx(edges, cv2.MORPH_OPEN, k_h))
    edges = cv2.subtract(edges, cv2.morphologyEx(edges, cv2.MORPH_OPEN, k_v))

    edges_clean = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, THICKEN_KERNEL),
        iterations=1
    )

    H, W = edges_clean.shape
    vis = frame.copy()
    skip_mask = np.zeros((H, W), dtype=bool)

    # -----------------------------
    # RING-SCAN + FARVE
    # -----------------------------
    for y in range(0, H, STEP):
        for x in range(0, W, STEP):

            if skip_mask[y, x]:
                continue
            if edges_clean[y, x] != 0:
                continue

            xs = x + ring_dx
            ys = y + ring_dy
            inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            xs, ys = xs[inside], ys[inside]

            if xs.size == 0:
                continue

            coverage = np.count_nonzero(edges_clean[ys, xs]) / xs.size
            if coverage < COVER_REQ:
                continue

            # -----------------------------
            # FARVEANALYSE
            # -----------------------------
            r_inner = RADIUS - 2
            x0, x1 = max(0, x - r_inner), min(W, x + r_inner + 1)
            y0, y1 = max(0, y - r_inner), min(H, y + r_inner + 1)

            roi = frame[y0:y1, x0:x1]
            if roi.size == 0:
                continue

            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
            cy, cx = y - y0, x - x0
            circle_mask = (xx - cx)**2 + (yy - cy)**2 <= r_inner*r_inner

            Hc = roi_hsv[..., 0][circle_mask]
            Sc = roi_hsv[..., 1][circle_mask]
            Vc = roi_hsv[..., 2][circle_mask]

            if Hc.size < 20:
                continue

            # ---------- ORANGE ----------
            orange_strong = (
                (Hc >= ORANGE_H_MIN) &
                (Hc <= ORANGE_H_MAX) &
                (Sc >= ORANGE_S_STRONG)
            )

            orange_weak = (
                (Hc >= ORANGE_H_MIN) &
                (Hc <= ORANGE_H_MAX) &
                (Sc >= ORANGE_S_WEAK) &
                (Vc >= ORANGE_V_MIN)
            )

            orange_ratio = np.mean(orange_strong | orange_weak)

            # ---------- WHITE ----------
            white_mask = (
                (Sc <= WHITE_S_MAX) |
                (Vc >= WHITE_V_MIN)
            )
            white_ratio = np.mean(white_mask)

            if orange_ratio >= COLOR_COVER_REQ:
                label = "ORANGE"
                draw_color = (0, 165, 255)

            elif white_ratio >= COLOR_COVER_REQ:
                # 🔑 override: hvid → orange hvis der er nok orange-nuancer
                if orange_ratio >= ORANGE_OVERRIDE_RATIO:
                    label = "ORANGE"
                    draw_color = (0, 165, 255)
                else:
                    label = "WHITE"
                    draw_color = (255, 255, 255)
            else:
                continue


            # -----------------------------
            # VISUALISERING
            # -----------------------------
            cv2.circle(vis, (x, y), RADIUS, draw_color, 2)
            cv2.putText(
                vis,
                label,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            # skip-område (performance + stabilitet)
            sx = x + skip_dx
            sy = y + skip_dy
            inside2 = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
            skip_mask[sy[inside2], sx[inside2]] = True

    # -----------------------------
    # VISNING
    # -----------------------------
    cv2.imshow("camera", frame)
    cv2.imshow("edges_clean", edges_clean)
    cv2.imshow("detected_balls", vis)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()