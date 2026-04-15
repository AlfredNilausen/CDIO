import cv2
import numpy as np


def make_board_mask(frame_shape, corners):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.array([
        corners["TL"],
        corners["TR"],
        corners["BR"],
        corners["BL"]
    ], dtype=np.int32)

    cv2.fillPoly(mask, [pts], 255)
    return mask


def make_inner_play_area_mask(frame_shape, corners, red_mask, margin_px=12):
    """
    Indre spilleområde:
    - board mask
    - lidt trukket ind fra kanten
    - rød ramme/kryds fjernes
    """
    board_mask = make_board_mask(frame_shape, corners)

    # Mindre erosion end før (25 var sandsynligvis for aggressiv)
    ksize = 2 * margin_px + 1
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    inner_mask = cv2.erode(board_mask, erode_kernel, iterations=1)

    # Fjern rød ramme/kryds
    red_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    red_expanded = cv2.dilate(red_mask, red_kernel, iterations=2)

    play_area = cv2.bitwise_and(inner_mask, cv2.bitwise_not(red_expanded))
    return play_area


def get_orange_ball_mask(frame, roi_mask=None):
    """
    Mere tolerant orange maske.
    Målet er først og fremmest at bolden SKAL være synlig i masken.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mere tolerant end før
    lower_orange = np.array([2, 40, 60])
    upper_orange = np.array([28, 255, 255])

    mask_hsv = cv2.inRange(hsv, lower_orange, upper_orange)

    # Ekstra farvefilter i BGR, men mildere end før
    b, g, r = cv2.split(frame)
    warm_cond = (
        (r.astype(np.int16) - b.astype(np.int16) > 20) &
        (g.astype(np.int16) - b.astype(np.int16) > 0) &
        (r > 70)
    )

    mask_warm = np.zeros_like(mask_hsv)
    mask_warm[warm_cond] = 255

    # Kombiner – men ikke for hårdt
    mask = cv2.bitwise_and(mask_hsv, mask_warm)

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def get_orange_ball_mask_hsv_only(frame, roi_mask=None):
    """
    Debug: kun HSV-versionen af orange
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([2, 40, 60])
    upper_orange = np.array([28, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    return mask


def get_white_ball_mask(frame, roi_mask=None):
    """
    Hvid bold:
    Brug en simpel og mere tolerant strategi:
    - lysstyrke i gray
    - top-hat til små lyse objekter
    - løs HSV
    OBS: vi kombinerer mere blødt end før
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Løs/off-white HSV
    lower_white_hsv = np.array([0, 0, 120])
    upper_white_hsv = np.array([180, 120, 255])
    mask_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)

    # Små lyse objekter
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
    _, mask_tophat = cv2.threshold(top_hat, 8, 255, cv2.THRESH_BINARY)

    # Direkte brightness
    _, mask_gray = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)

    # VIGTIGT:
    # Før brugte vi AND mellem flere hårde filtre.
    # Nu bruger vi OR mellem bright-kilderne og derefter AND med loose HSV.
    bright_mask = cv2.bitwise_or(mask_tophat, mask_gray)
    mask = cv2.bitwise_and(mask_hsv, bright_mask)

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def get_white_ball_bright_only(frame, roi_mask=None):
    """
    Debug: kun brightness-del til hvid bold
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
    _, mask_tophat = cv2.threshold(top_hat, 8, 255, cv2.THRESH_BINARY)

    _, mask_gray = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)

    bright_mask = cv2.bitwise_or(mask_tophat, mask_gray)

    if roi_mask is not None:
        bright_mask = cv2.bitwise_and(bright_mask, roi_mask)

    return bright_mask


def contour_circularity(cnt):
    area = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt, True)

    if per == 0:
        return 0.0

    return 4.0 * np.pi * area / (per * per)


def find_ball_candidates(mask, min_area=15, max_area=1200, min_circularity=0.45):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    balls = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        circ = contour_circularity(cnt)
        if circ < min_circularity:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)

        # Mere tolerant radius-range
        if radius < 3 or radius > 20:
            continue

        balls.append({
            "center": (int(x), int(y)),
            "radius": int(radius),
            "area": area,
            "circularity": circ,
            "contour": cnt
        })

    return balls