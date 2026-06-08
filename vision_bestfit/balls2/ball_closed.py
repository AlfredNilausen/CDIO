import cv2
import numpy as np
from dataclasses import dataclass


# =========================
# CONFIGS
# =========================
@dataclass
class BallEdgesConfig:
    # DINE "GODE" DEFAULTS (fra din nuværende ball_edges)
    canny_t1: int = 110
    canny_t2: int = 300
    blur_k: int = 31
    blur_sigma: int = 1
    dilate_k: int = 3
    dilate_it: int = 2


@dataclass
class BallClosedConfig:
    # Closing (binder gaps i kanter)
    close_k: int = 9          # kernel size (ulige)
    close_it: int = 2         # iterations

    # Fill (lav lukket kontur til en fyldt maske)
    fill: int = 1             # 1=ON, 0=OFF

    # Ryd støj / filtrering
    min_area: int = 500       # konturer mindre end dette ignoreres

    # Optional: lidt "pænere" maske efter fill
    post_open_k: int = 0      # 0 = OFF (ellers ulige)
    post_open_it: int = 1


# =========================
# HELPERS
# =========================
def _odd(n: int) -> int:
    n = int(n)
    n = max(1, n)
    return n if (n % 2 == 1) else n + 1


# =========================
# BALL EDGE LOGIC
# =========================
def compute_ball_edges(frame, cfg: BallEdgesConfig):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur (ensure odd kernel)
    k = max(1, cfg.blur_k | 1)
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), cfg.blur_sigma)

    # Canny
    edges = cv2.Canny(gray, cfg.canny_t1, cfg.canny_t2)

    # Dilate edges
    dk = max(1, cfg.dilate_k | 1)
    if dk > 1 and cfg.dilate_it > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
        edges = cv2.dilate(edges, kernel, iterations=cfg.dilate_it)

    return edges


# =========================
# CLOSE + (OPTIONAL) FILL
# =========================
def close_and_fill(edges: np.ndarray, cfg: BallClosedConfig):
    # 1) Closing på edges: binder små huller/gaps i kanter
    ck = _odd(cfg.close_k)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=max(0, cfg.close_it))

    # 2) Lav en fyldt maske (hvis ønsket)
    if cfg.fill:
        filled = np.zeros_like(edges)
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= cfg.min_area:
                cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)

        # 3) Optional: post-open for at fjerne små "støjklumper"
        if cfg.post_open_k and cfg.post_open_k > 1 and cfg.post_open_it > 0:
            ok = _odd(cfg.post_open_k)
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
            filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, open_kernel, iterations=cfg.post_open_it)

        return closed_edges, filled

    return closed_edges, None


# =========================
# LIVE TEST / TUNER
# =========================
def run_ball_closed_tuner(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("closed_edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("filled_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("tuner", cv2.WINDOW_NORMAL)

    # Start fra defaults
    base_edges = BallEdgesConfig()
    base_closed = BallClosedConfig()

    # -------- TRACKBARS: EDGES --------
    cv2.createTrackbar("canny_t1", "tuner", base_edges.canny_t1, 300, lambda x: None)
    cv2.createTrackbar("canny_t2", "tuner", base_edges.canny_t2, 300, lambda x: None)
    cv2.createTrackbar("blur_k", "tuner", base_edges.blur_k, 31, lambda x: None)
    cv2.createTrackbar("blur_sigma", "tuner", base_edges.blur_sigma, 10, lambda x: None)
    cv2.createTrackbar("dilate_k", "tuner", base_edges.dilate_k, 21, lambda x: None)
    cv2.createTrackbar("dilate_it", "tuner", base_edges.dilate_it, 10, lambda x: None)

    # -------- TRACKBARS: CLOSING + FILL --------
    cv2.createTrackbar("close_k", "tuner", base_closed.close_k, 51, lambda x: None)
    cv2.createTrackbar("close_it", "tuner", base_closed.close_it, 10, lambda x: None)
    cv2.createTrackbar("fill", "tuner", base_closed.fill, 1, lambda x: None)
    cv2.createTrackbar("min_area", "tuner", base_closed.min_area, 20000, lambda x: None)
    cv2.createTrackbar("post_open_k", "tuner", base_closed.post_open_k, 51, lambda x: None)
    cv2.createTrackbar("post_open_it", "tuner", base_closed.post_open_it, 10, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Læs edges cfg fra sliders
        edges_cfg = BallEdgesConfig(
            canny_t1=cv2.getTrackbarPos("canny_t1", "tuner"),
            canny_t2=cv2.getTrackbarPos("canny_t2", "tuner"),
            blur_k=cv2.getTrackbarPos("blur_k", "tuner"),
            blur_sigma=cv2.getTrackbarPos("blur_sigma", "tuner"),
            dilate_k=cv2.getTrackbarPos("dilate_k", "tuner"),
            dilate_it=cv2.getTrackbarPos("dilate_it", "tuner"),
        )

        # Læs close cfg fra sliders
        closed_cfg = BallClosedConfig(
            close_k=cv2.getTrackbarPos("close_k", "tuner"),
            close_it=cv2.getTrackbarPos("close_it", "tuner"),
            fill=cv2.getTrackbarPos("fill", "tuner"),
            min_area=cv2.getTrackbarPos("min_area", "tuner"),
            post_open_k=cv2.getTrackbarPos("post_open_k", "tuner"),
            post_open_it=cv2.getTrackbarPos("post_open_it", "tuner"),
        )

        edges = compute_ball_edges(frame, edges_cfg)
        closed_edges, filled = close_and_fill(edges, closed_cfg)

        # Debug overlay (så du kan se, at sliders faktisk ændrer)
        dbg = frame.copy()
        cv2.putText(
            dbg,
            f"Canny {edges_cfg.canny_t1}/{edges_cfg.canny_t2} | Blur k={edges_cfg.blur_k} s={edges_cfg.blur_sigma} | Dilate k={edges_cfg.dilate_k} it={edges_cfg.dilate_it}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            dbg,
            f"Close k={_odd(closed_cfg.close_k)} it={closed_cfg.close_it} | Fill={closed_cfg.fill} | min_area={closed_cfg.min_area} | post_open={closed_cfg.post_open_k}x{closed_cfg.post_open_it}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("camera", dbg)
        cv2.imshow("edges", edges)
        cv2.imshow("closed_edges", closed_edges)

        if filled is None:
            cv2.imshow("filled_mask", np.zeros_like(edges))
        else:
            cv2.imshow("filled_mask", filled)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("s"):
            print("\n=== CURRENT BALL EDGES CONFIG ===")
            print(edges_cfg)
            print("=== CURRENT BALL CLOSED CONFIG ===")
            print(closed_cfg)

    cap.release()
    cv2.destroyAllWindows()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run_ball_closed_tuner()
