# white_intersection.py
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from white_mask import compute_white_mask, WhiteMaskConfig
from ball_edges import BallEdgesConfig
from ball_closed import compute_ball_closed, BallClosedConfig



from dataclasses import dataclass, field
@dataclass
class WhiteIntersectionConfig:
    white: WhiteMaskConfig = field(default_factory=WhiteMaskConfig)
    edges: BallEdgesConfig = field(default_factory=BallEdgesConfig)
    closed: BallClosedConfig = field(default_factory=BallClosedConfig)

    erode_after_intersection: bool = False
    erode_kernel_size: int = 3
    erode_iterations: int = 1


def _ensure_odd(n: int, minimum: int = 1) -> int:
    n = int(n)
    n = max(minimum, n)
    return n if n % 2 == 1 else n + 1


def compute_white_intersection(
    frame: np.ndarray,
    config: Optional[WhiteIntersectionConfig] = None,
    return_debug: bool = True
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Beholder kun pixels der er hvide i både:
      - white_mask (farve)
      - ball_closed (lukket geometri)

    Args:
        frame: BGR frame
        config: WhiteIntersectionConfig (valgfri)
        return_debug:
            True  -> returnerer (intersection, white_mask, closed_filled, edges, edges_closed)
            False -> returnerer kun intersection

    Returns:
        intersection (og evt debug outputs)
    """
    if config is None:
        config = WhiteIntersectionConfig()

    # 1) Farve-maske (hvid)
    wmask = compute_white_mask(frame, config.white)

    # 2) Geometri (edges -> closed filled)
    closed_filled, edges, edges_closed = compute_ball_closed(
        frame=frame,
        edges=None,
        edges_config=config.edges,
        closed_config=config.closed,
        return_debug=True
    )

    # 3) Intersection (AND)
    intersection = cv2.bitwise_and(wmask, closed_filled)

    # (valgfrit) lidt erosion for at fjerne tynde “haler”/støj
    if config.erode_after_intersection and config.erode_iterations > 0:
        k = _ensure_odd(config.erode_kernel_size, minimum=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        intersection = cv2.erode(intersection, kernel, iterations=int(config.erode_iterations))

    if return_debug:
        return intersection, wmask, closed_filled, edges, edges_closed
    return intersection


def compute_white_intersection_with_params(
    frame: np.ndarray,
    # --- white_mask parametre (de vigtigste) ---
    white_s_max: int = 60,
    white_v_min: int = 130,

    # --- edges parametre (de vigtigste) ---
    canny_t1: int = 50,
    canny_t2: int = 150,
    remove_lines: bool = True,

    # --- closed parametre ---
    close_kernel_size: int = 5,
    close_iterations: int = 2,
    min_area: int = 80,

    # --- post ---
    erode_after_intersection: bool = False,
    erode_kernel_size: int = 3,
    erode_iterations: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: tweak uden at oprette configs.
    Returnerer altid debug: (intersection, white_mask, closed_filled, edges, edges_closed)
    """
    cfg = WhiteIntersectionConfig(
        white=WhiteMaskConfig(white_s_max=white_s_max, white_v_min=white_v_min),
        edges=BallEdgesConfig(canny_t1=canny_t1, canny_t2=canny_t2, remove_lines=remove_lines),
        closed=BallClosedConfig(close_kernel_size=close_kernel_size, close_iterations=close_iterations, min_area=min_area),
        erode_after_intersection=erode_after_intersection,
        erode_kernel_size=erode_kernel_size,
        erode_iterations=erode_iterations
    )

    return compute_white_intersection(frame, cfg, return_debug=True)


# ----------------------------
# OPTIONAL: Live tuner (webcam / video)
# ----------------------------
def run_tuner(
    source: Union[int, str] = 0,
    width: int = 1280,
    height: int = 720,
    use_dshow: bool = True
) -> None:
    """
    Live tuner med trackbars.
    source:
      - 0,1,2,... for webcam
      - "path/to/video.mp4" for videofil
    ESC lukker.
    """
    backend = cv2.CAP_DSHOW if (use_dshow and isinstance(source, int)) else 0
    cap = cv2.VideoCapture(source, backend)

    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("white_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edges_closed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ball_closed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("white_intersection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("tuner", cv2.WINDOW_NORMAL)

    # --- Trackbars: White mask ---
    cv2.createTrackbar("white_v_min", "tuner", 130, 255, lambda x: None)
    cv2.createTrackbar("white_s_max", "tuner", 60, 255, lambda x: None)

    # --- Trackbars: Edges (Canny + line removal) ---
    cv2.createTrackbar("canny_t1", "tuner", 50, 300, lambda x: None)
    cv2.createTrackbar("canny_t2", "tuner", 150, 400, lambda x: None)
    cv2.createTrackbar("remove_lines", "tuner", 1, 1, lambda x: None)

    # --- Trackbars: Closed fill ---
    cv2.createTrackbar("close_k", "tuner", 5, 31, lambda x: None)
    cv2.createTrackbar("close_it", "tuner", 2, 10, lambda x: None)
    cv2.createTrackbar("min_area", "tuner", 80, 2000, lambda x: None)

    # --- Trackbars: Post ---
    cv2.createTrackbar("erode_on", "tuner", 0, 1, lambda x: None)
    cv2.createTrackbar("erode_k", "tuner", 3, 21, lambda x: None)
    cv2.createTrackbar("erode_it", "tuner", 1, 10, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            # hvis videofil er slut, loop eller stop (her: loop)
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        white_v_min = cv2.getTrackbarPos("white_v_min", "tuner")
        white_s_max = cv2.getTrackbarPos("white_s_max", "tuner")

        canny_t1 = cv2.getTrackbarPos("canny_t1", "tuner")
        canny_t2 = cv2.getTrackbarPos("canny_t2", "tuner")
        remove_lines = bool(cv2.getTrackbarPos("remove_lines", "tuner"))

        close_k = cv2.getTrackbarPos("close_k", "tuner")
        close_it = cv2.getTrackbarPos("close_it", "tuner")
        min_area = cv2.getTrackbarPos("min_area", "tuner")

        erode_on = bool(cv2.getTrackbarPos("erode_on", "tuner"))
        erode_k = cv2.getTrackbarPos("erode_k", "tuner")
        erode_it = cv2.getTrackbarPos("erode_it", "tuner")

        cfg = WhiteIntersectionConfig(
            white=WhiteMaskConfig(white_s_max=white_s_max, white_v_min=white_v_min),
            edges=BallEdgesConfig(canny_t1=canny_t1, canny_t2=canny_t2, remove_lines=remove_lines),
            closed=BallClosedConfig(close_kernel_size=close_k, close_iterations=close_it, min_area=min_area),
            erode_after_intersection=erode_on,
            erode_kernel_size=erode_k,
            erode_iterations=erode_it
        )

        intersection, wmask, closed_filled, edges, edges_closed = compute_white_intersection(
            frame, cfg, return_debug=True
        )

        cv2.imshow("camera", frame)
        cv2.imshow("white_mask", wmask)
        cv2.imshow("edges", edges)
        cv2.imshow("edges_closed", edges_closed)
        cv2.imshow("ball_closed", closed_filled)
        cv2.imshow("white_intersection", intersection)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------
# TEST / DEMO
# -------------------------------------------------
if __name__ == "__main__":
    # Webcam:
    run_tuner(source=0)

    # Video eksempel:
    # run_tuner(source="testvideos/white_ball.mp4", use_dshow=False)
