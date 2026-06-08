import cv2
import numpy as np

from white_mask import compute_white_mask, WhiteMaskConfig
from ball_edges import compute_ball_edges, BallEdgesConfig


def circle_white_ratio(white_mask: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Andel af hvide pixels inde i en cirkel."""
    h, w = white_mask.shape[:2]
    if r <= 0:
        return 0.0

    x0, y0 = max(0, cx - r), max(0, cy - r)
    x1, y1 = min(w, cx + r + 1), min(h, cy + r + 1)

    roi = white_mask[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.circle(mask, (cx - x0, cy - y0), r, 255, thickness=-1)

    inside = cv2.bitwise_and(roi, mask)
    white_pixels = cv2.countNonZero(inside)
    total_pixels = cv2.countNonZero(mask)
    return white_pixels / float(total_pixels) if total_pixels > 0 else 0.0


def run_edges_first_detector(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    use_dshow: bool = True,
):
    backend = cv2.CAP_DSHOW if use_dshow else 0
    cap = cv2.VideoCapture(camera_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("white_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("tuner", cv2.WINDOW_NORMAL)

    # --- Defaults: white mask (kun de vigtigste tweaks i denne tuner) ---
    white_base = WhiteMaskConfig()
    # OBS: dine seneste setups var mere "permissive". Start med base, tune her.
    cv2.createTrackbar("white_s_max", "tuner", white_base.white_s_max, 255, lambda x: None)
    cv2.createTrackbar("white_v_min", "tuner", white_base.white_v_min, 255, lambda x: None)

    # --- Defaults: ball_edges (fra din ball_edges.py defaults) ---
    edges_base = BallEdgesConfig()
    cv2.createTrackbar("canny_t1", "tuner", edges_base.canny_t1, 300, lambda x: None)
    cv2.createTrackbar("canny_t2", "tuner", edges_base.canny_t2, 300, lambda x: None)
    cv2.createTrackbar("blur_k", "tuner", edges_base.blur_k, 31, lambda x: None)
    cv2.createTrackbar("blur_sigma", "tuner", edges_base.blur_sigma, 10, lambda x: None)
    cv2.createTrackbar("dilate_k", "tuner", edges_base.dilate_k, 21, lambda x: None)
    cv2.createTrackbar("dilate_it", "tuner", edges_base.dilate_it, 10, lambda x: None)

    # --- Circle detection (Hough) defaults for r ≈ 7-9 px ---
    cv2.createTrackbar("r_min", "tuner", 6, 30, lambda x: None)
    cv2.createTrackbar("r_max", "tuner", 11, 40, lambda x: None)
    cv2.createTrackbar("minDist", "tuner", 18, 80, lambda x: None)   # ~ 2*r (start 18)
    cv2.createTrackbar("param2", "tuner", 18, 80, lambda x: None)    # accumulator threshold (start 18)

    # --- White classification threshold ---
    cv2.createTrackbar("white_ratio(%)", "tuner", 60, 100, lambda x: None)  # start 60%

    # --- Debug toggles ---
    cv2.createTrackbar("show_nonwhite", "tuner", 1, 1, lambda x: None)      # show non-white circles too

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Read tuned configs
        white_cfg = WhiteMaskConfig(
            white_h_min=white_base.white_h_min,
            white_h_max=white_base.white_h_max,
            white_s_min=white_base.white_s_min,
            white_s_max=cv2.getTrackbarPos("white_s_max", "tuner"),
            white_v_min=cv2.getTrackbarPos("white_v_min", "tuner"),
            white_v_max=white_base.white_v_max,
            orange_h_min=0, orange_h_max=0,
            orange_s_min=0, orange_s_max=0,
            orange_v_min=0, orange_v_max=0,
            kernel_size=white_base.kernel_size,
            open_iterations=white_base.open_iterations,
            close_iterations=white_base.close_iterations,
        )

        edges_cfg = BallEdgesConfig(
            canny_t1=cv2.getTrackbarPos("canny_t1", "tuner"),
            canny_t2=cv2.getTrackbarPos("canny_t2", "tuner"),
            blur_k=cv2.getTrackbarPos("blur_k", "tuner"),
            blur_sigma=cv2.getTrackbarPos("blur_sigma", "tuner"),
            dilate_k=cv2.getTrackbarPos("dilate_k", "tuner"),
            dilate_it=cv2.getTrackbarPos("dilate_it", "tuner"),
        )

        r_min = cv2.getTrackbarPos("r_min", "tuner")
        r_max = cv2.getTrackbarPos("r_max", "tuner")
        if r_max < r_min:
            r_max = r_min

        minDist = max(1, cv2.getTrackbarPos("minDist", "tuner"))
        param2 = max(1, cv2.getTrackbarPos("param2", "tuner"))
        white_ratio_thr = cv2.getTrackbarPos("white_ratio(%)", "tuner") / 100.0
        show_nonwhite = bool(cv2.getTrackbarPos("show_nonwhite", "tuner"))

        # Compute masks
        wm = compute_white_mask(frame, white_cfg)
        edges = compute_ball_edges(frame, edges_cfg)

        # Hough wants a gray image (it does its own edge processing internally)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Let blur_k drive pre-blur consistency
        k = max(1, edges_cfg.blur_k | 1)
        if k > 1:
            gray = cv2.GaussianBlur(gray, (k, k), edges_cfg.blur_sigma)

        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=minDist,
            param1=edges_cfg.canny_t2,   # use canny_t2 as strong threshold proxy
            param2=param2,
            minRadius=r_min,
            maxRadius=r_max,
        )

        dbg = frame.copy()
        white_balls = []
        nonwhite_balls = []

        if circles is not None:
            circles = np.round(circles[0]).astype(int)

            # Simple de-dup: keep circles that aren't almost identical
            kept = []
            for (cx, cy, r) in circles:
                ok = True
                for (kx, ky, kr) in kept:
                    if (cx - kx) ** 2 + (cy - ky) ** 2 < 9 and abs(r - kr) <= 1:
                        ok = False
                        break
                if ok:
                    kept.append((cx, cy, r))

            for (cx, cy, r) in kept:
                ratio = circle_white_ratio(wm, cx, cy, r)

                if ratio >= white_ratio_thr:
                    white_balls.append((cx, cy, r, ratio))
                else:
                    nonwhite_balls.append((cx, cy, r, ratio))

        # Draw results
        # White balls: cyan/yellow
        for (cx, cy, r, ratio) in white_balls:
            cv2.circle(dbg, (cx, cy), r, (0, 255, 255), 2)
            cv2.circle(dbg, (cx, cy), 2, (0, 255, 255), -1)
            cv2.putText(dbg, f"W {ratio*100:.0f}%", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Non-white circles (optional): red
        if show_nonwhite:
            for (cx, cy, r, ratio) in nonwhite_balls:
                cv2.circle(dbg, (cx, cy), r, (0, 0, 255), 1)
                cv2.putText(dbg, f"{ratio*100:.0f}%", (cx + 6, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.putText(
            dbg,
            f"White balls: {len(white_balls)} | Nonwhite circles: {len(nonwhite_balls)} | r[{r_min}-{r_max}] minDist={minDist} param2={param2} white_thr={white_ratio_thr:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        cv2.imshow("camera", dbg)
        cv2.imshow("white_mask", wm)
        cv2.imshow("edges", edges)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("s"):
            print("\n=== CURRENT SETTINGS (EDGES-FIRST) ===")
            print("WhiteMask:", {"white_s_max": white_cfg.white_s_max, "white_v_min": white_cfg.white_v_min})
            print("BallEdges:", edges_cfg)
            print("Hough:", {"r_min": r_min, "r_max": r_max, "minDist": minDist, "param2": param2})
            print("white_ratio_thr:", white_ratio_thr)
            print("white_balls (cx,cy,r,ratio):", white_balls)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_edges_first_detector()
