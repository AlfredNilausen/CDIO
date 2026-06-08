import cv2
import numpy as np
from dataclasses import dataclass


# =========================
# CONFIG (DEFAULTS = dine gode settings)
# =========================
@dataclass
class BallEdgesConfig:
    canny_t1: int = 110
    canny_t2: int = 300
    blur_k: int = 31
    blur_sigma: int = 1
    dilate_k: int = 3
    dilate_it: int = 2


# =========================
# BALL EDGE LOGIC
# =========================
def compute_ball_edges(frame, cfg: BallEdgesConfig):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur (ensure odd kernel)
    k = max(1, cfg.blur_k | 1)  # |1 gør ulige
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), cfg.blur_sigma)

    # Canny
    edges = cv2.Canny(gray, cfg.canny_t1, cfg.canny_t2)

    # Dilate edges
    dk = max(1, cfg.dilate_k | 1)  # |1 gør ulige
    if dk > 1 and cfg.dilate_it > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
        edges = cv2.dilate(edges, kernel, iterations=cfg.dilate_it)

    return edges


# =========================
# LIVE TEST / TUNER
# =========================
def run_ball_edges_test(
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
    cv2.namedWindow("ball_edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("tuner", cv2.WINDOW_NORMAL)

    # ✅ Start fra defaults
    base = BallEdgesConfig()

    # -------- TRACKBARS (starter på base-values) --------
    cv2.createTrackbar("canny_t1", "tuner", base.canny_t1, 300, lambda x: None)
    cv2.createTrackbar("canny_t2", "tuner", base.canny_t2, 300, lambda x: None)

    cv2.createTrackbar("blur_k", "tuner", base.blur_k, 31, lambda x: None)
    cv2.createTrackbar("blur_sigma", "tuner", base.blur_sigma, 10, lambda x: None)

    cv2.createTrackbar("dilate_k", "tuner", base.dilate_k, 21, lambda x: None)
    cv2.createTrackbar("dilate_it", "tuner", base.dilate_it, 5, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cfg = BallEdgesConfig(
            canny_t1=cv2.getTrackbarPos("canny_t1", "tuner"),
            canny_t2=cv2.getTrackbarPos("canny_t2", "tuner"),
            blur_k=cv2.getTrackbarPos("blur_k", "tuner"),
            blur_sigma=cv2.getTrackbarPos("blur_sigma", "tuner"),
            dilate_k=cv2.getTrackbarPos("dilate_k", "tuner"),
            dilate_it=cv2.getTrackbarPos("dilate_it", "tuner"),
        )

        edges = compute_ball_edges(frame, cfg)

        # Debug overlay: viser live værdier på kamera-billedet
        dbg = frame.copy()
        cv2.putText(
            dbg,
            f"Canny {cfg.canny_t1}/{cfg.canny_t2} | Blur k={cfg.blur_k} s={cfg.blur_sigma} | Dilate k={cfg.dilate_k} it={cfg.dilate_it}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

        cv2.imshow("camera", dbg)
        cv2.imshow("ball_edges", edges)

        key = cv2.waitKey(1) & 0xFF  # ✅ vigtig: & ikke &amp;
        if key == 27:  # ESC
            break
        elif key == ord("s"):
            print("\n=== CURRENT BALL EDGES CONFIG ===")
            print(cfg)

    cap.release()
    cv2.destroyAllWindows()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run_ball_edges_test()
