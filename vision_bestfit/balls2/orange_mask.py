import cv2
import numpy as np

WINDOW = "tuner_orange_mask"

DEFAULT = {
    "h_min": 3, "s_min": 65, "v_min": 65,
    "h_max": 32, "s_max": 255, "v_max": 255,
    "kernel": 5, "open_it": 2, "close_it": 2,
}

def _tb(n):
    try: return cv2.getTrackbarPos(n, WINDOW)
    except cv2.error: return DEFAULT[n]

def setup_orange_mask_tuner():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("h_min", WINDOW, 3, 179, lambda x:None)
    cv2.createTrackbar("s_min", WINDOW, 65, 255, lambda x:None)
    cv2.createTrackbar("v_min", WINDOW, 65, 255, lambda x:None)
    cv2.createTrackbar("h_max", WINDOW, 32, 179, lambda x:None)
    cv2.createTrackbar("s_max", WINDOW, 255, 255, lambda x:None)
    cv2.createTrackbar("v_max", WINDOW, 255, 255, lambda x:None)
    cv2.createTrackbar("kernel", WINDOW, 5, 25, lambda x:None)
    cv2.createTrackbar("open_it", WINDOW, 2, 10, lambda x:None)
    cv2.createTrackbar("close_it", WINDOW, 2, 10, lambda x:None)

def compute_orange_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([_tb("h_min"), _tb("s_min"), _tb("v_min")]),
        np.array([_tb("h_max"), _tb("s_max"), _tb("v_max")])
    )
    k = max(1, _tb("kernel") | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    if _tb("open_it")>0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=_tb("open_it"))
    if _tb("close_it")>0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=_tb("close_it"))
    return mask
