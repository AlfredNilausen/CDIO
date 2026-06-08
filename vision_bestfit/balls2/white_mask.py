# white_mask.py
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# =========================
# CONFIG
# =========================
@dataclass
class WhiteMaskConfig:
    # WHITE HSV
    white_h_min: int = 0
    white_h_max: int = 179
    white_s_min: int = 0
    white_s_max: int = 65
    white_v_min: int = 207
    white_v_max: int = 255

    # ORANGE (disabled by default)
    orange_h_min: int = 0
    orange_h_max: int = 0
    orange_s_min: int = 0
    orange_s_max: int = 0
    orange_v_min: int = 0
    orange_v_max: int = 0

    # MORPH
    kernel_size: int = 5
    open_iterations: int = 1
    close_iterations: int = 2


# =========================
# HELPERS
# =========================
def _ensure_odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


# =========================
# CORE MASK
# =========================
def compute_white_mask(frame: np.ndarray, config: Optional[WhiteMaskConfig] = None) -> np.ndarray:
    if config is None:
        config = WhiteMaskConfig()

    # Clamp (sikkerhed ift. sliders)
    config.white_h_min = _clamp(config.white_h_min, 0, 179)
    config.white_h_max = _clamp(config.white_h_max, 0, 179)
    config.white_s_min = _clamp(config.white_s_min, 0, 255)
    config.white_s_max = _clamp(config.white_s_max, 0, 255)
    config.white_v_min = _clamp(config.white_v_min, 0, 255)
    config.white_v_max = _clamp(config.white_v_max, 0, 255)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([config.white_h_min, config.white_s_min, config.white_v_min], dtype=np.uint8)
    upper_white = np.array([config.white_h_max, config.white_s_max, config.white_v_max], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Orange removal
    if any([
        config.orange_h_min, config.orange_h_max,
        config.orange_s_min, config.orange_s_max,
        config.orange_v_min, config.orange_v_max
    ]):
        lower_orange = np.array([config.orange_h_min, config.orange_s_min, config.orange_v_min], dtype=np.uint8)
        upper_orange = np.array([config.orange_h_max, config.orange_s_max, config.orange_v_max], dtype=np.uint8)
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(orange_mask))

    # Morphology
    k = _ensure_odd(config.kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    if config.open_iterations > 0:
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=config.open_iterations)

    if config.close_iterations > 0:
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=config.close_iterations)

    return white_mask


# =========================
# FULL TUNER
# =========================
def run_camera_tuner(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    use_dshow: bool = True,
) -> None:

    backend = cv2.CAP_DSHOW if use_dshow else 0
    cap = cv2.VideoCapture(camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("white_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("tuner", cv2.WINDOW_NORMAL)

    # Start fra dine defaults
    base = WhiteMaskConfig()

    # -------- WHITE HSV --------
    cv2.createTrackbar("white_h_min", "tuner", base.white_h_min, 179, lambda x: None)
    cv2.createTrackbar("white_h_max", "tuner", base.white_h_max, 179, lambda x: None)
    cv2.createTrackbar("white_s_min", "tuner", base.white_s_min, 255, lambda x: None)
    cv2.createTrackbar("white_s_max", "tuner", base.white_s_max, 255, lambda x: None)
    cv2.createTrackbar("white_v_min", "tuner", base.white_v_min, 255, lambda x: None)
    cv2.createTrackbar("white_v_max", "tuner", base.white_v_max, 255, lambda x: None)

    # -------- ORANGE HSV --------
    # base har orange disabled -> start OFF
    cv2.createTrackbar("remove_orange", "tuner", 0, 1, lambda x: None)
    cv2.createTrackbar("orange_h_min", "tuner", 8, 179, lambda x: None)
    cv2.createTrackbar("orange_h_max", "tuner", 25, 179, lambda x: None)
    cv2.createTrackbar("orange_s_min", "tuner", 120, 255, lambda x: None)
    cv2.createTrackbar("orange_s_max", "tuner", 255, 255, lambda x: None)
    cv2.createTrackbar("orange_v_min", "tuner", 120, 255, lambda x: None)
    cv2.createTrackbar("orange_v_max", "tuner", 255, 255, lambda x: None)

    # -------- MORPH --------
    cv2.createTrackbar("kernel_size", "tuner", base.kernel_size, 31, lambda x: None)
    cv2.createTrackbar("open_iter", "tuner", base.open_iterations, 10, lambda x: None)
    cv2.createTrackbar("close_iter", "tuner", base.close_iterations, 10, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cfg = WhiteMaskConfig(
            white_h_min=cv2.getTrackbarPos("white_h_min", "tuner"),
            white_h_max=cv2.getTrackbarPos("white_h_max", "tuner"),
            white_s_min=cv2.getTrackbarPos("white_s_min", "tuner"),
            white_s_max=cv2.getTrackbarPos("white_s_max", "tuner"),
            white_v_min=cv2.getTrackbarPos("white_v_min", "tuner"),
            white_v_max=cv2.getTrackbarPos("white_v_max", "tuner"),
            kernel_size=cv2.getTrackbarPos("kernel_size", "tuner"),
            open_iterations=cv2.getTrackbarPos("open_iter", "tuner"),
            close_iterations=cv2.getTrackbarPos("close_iter", "tuner"),
        )

        if cv2.getTrackbarPos("remove_orange", "tuner") == 1:
            cfg.orange_h_min = cv2.getTrackbarPos("orange_h_min", "tuner")
            cfg.orange_h_max = cv2.getTrackbarPos("orange_h_max", "tuner")
            cfg.orange_s_min = cv2.getTrackbarPos("orange_s_min", "tuner")
            cfg.orange_s_max = cv2.getTrackbarPos("orange_s_max", "tuner")
            cfg.orange_v_min = cv2.getTrackbarPos("orange_v_min", "tuner")
            cfg.orange_v_max = cv2.getTrackbarPos("orange_v_max", "tuner")
        else:
            cfg.orange_h_min = cfg.orange_h_max = 0
            cfg.orange_s_min = cfg.orange_s_max = 0
            cfg.orange_v_min = cfg.orange_v_max = 0

        # ✅ VIGTIGT: brug cfg fra sliders
        mask = compute_white_mask(frame, cfg)

        # Optional: vis værdier på kamera-billedet, så du kan se det ændrer sig
        dbg = frame.copy()
        cv2.putText(
            dbg,
            f"H[{cfg.white_h_min}-{cfg.white_h_max}] S[{cfg.white_s_min}-{cfg.white_s_max}] V[{cfg.white_v_min}-{cfg.white_v_max}]",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("camera", dbg)
        cv2.imshow("white_mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("s"):
            print("\n=== CURRENT CONFIG ===")
            print(cfg)

    cap.release()
    cv2.destroyAllWindows()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run_camera_tuner()
