import cv2
from ball_edges import BallEdgesConfig
from ball_closed import BallClosedConfig

WINDOW = "ball_tuner"

def _tb(name, default):
    try:
        return cv2.getTrackbarPos(name, WINDOW)
    except cv2.error:
        return default

def setup_ball_tuner():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("canny_t1", WINDOW, 50, 300, lambda x: None)
    cv2.createTrackbar("canny_t2", WINDOW, 150, 400, lambda x: None)
    cv2.createTrackbar("close_k", WINDOW, 5, 31, lambda x: None)
    cv2.createTrackbar("close_it", WINDOW, 2, 10, lambda x: None)
    cv2.createTrackbar("min_area", WINDOW, 80, 3000, lambda x: None)
    cv2.createTrackbar("max_area", WINDOW, 5000, 10000, lambda x: None)

def get_ball_edges_config():
    return BallEdgesConfig(
        canny_t1=_tb("canny_t1", 50),
        canny_t2=_tb("canny_t2", 150),
    )

def get_ball_closed_config():
    return BallClosedConfig(
        close_k=_tb("close_k", 5),
        close_it=_tb("close_it", 2),
        min_area=_tb("min_area", 80),
        max_area=_tb("max_area", 5000),
    )
