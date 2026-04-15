import cv2
import numpy as np


def make_board_mask(frame_shape, corners):
    """
    Fyldt maske for hele banen ud fra hjørnerne.
    corners = {"TL":..., "TR":..., "BL":..., "BR":...}
    """
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


def make_play_area_mask(frame_shape, corners, red_mask, margin_px=18):
    """
    Indre søgeområde:
    - starter fra banens polygon
    - trækker sig lidt ind fra kanten
    - fjerner rød ramme + rødt kryds
    """
    board_mask = make_board_mask(frame_shape, corners)

    # Træk banen lidt indad
    ksize = 2 * margin_px + 1
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    inner_mask = cv2.erode(board_mask, erode_kernel, iterations=1)

    # Fjern rød maske (ramme + kryds)
    red_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    red_expanded = cv2.dilate(red_mask, red_kernel, iterations=2)

    play_area = cv2.bitwise_and(inner_mask, cv2.bitwise_not(red_expanded))
    return play_area


def get_orange_ball_mask(frame, roi_mask=None):
    """
    Orange bold:
    HSV-baseret maske til bordtennisbold i dette setup.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Relativt tolerant orange
    lower_orange = np.array([4, 90, 70])
    upper_orange = np.array([24, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def get_white_ball_mask(frame, roi_mask=None):
    """
    Hvide bolde:
    Kombination af:
    - høj lysstyrke
    - nogenlunde neutral farve
    - små lyse objekter (top-hat)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) HSV: hvid/off-white
    # høj value, lav/moderat saturation
    lower_white_hsv = np.array([0, 0, 150])
    upper_white_hsv = np.array([180, 70, 255])
    mask_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)

    # 2) LAB: nogenlunde neutral farve
    L, A, B = cv2.split(lab)

    neutral_mask = (
        (L > 160) &
        (np.abs(A.astype(np.int16) - 128) < 18) &
        (np.abs(B.astype(np.int16) - 128) < 24)
    )

    mask_lab = np.zeros_like(L, dtype=np.uint8)
    mask_lab[neutral_mask] = 255

    # 3) Top-hat for små lyse objekter
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
    _, mask_tophat = cv2.threshold(top_hat, 8, 255, cv2.THRESH_BINARY)

    # 4) Direkte brightness threshold
    _, mask_gray = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    bright_mask = cv2.bitwise_or(mask_tophat, mask_gray)

    # Kombinér
    mask = cv2.bitwise_and(mask_hsv, mask_lab)
    mask = cv2.bitwise_and(mask, bright_mask)

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def contour_circularity(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        return 0.0

    return 4.0 * np.pi * area / (perimeter * perimeter)


def find_ball_candidates(mask, min_area=20, max_area=1200, min_circularity=0.50):
    """
    Finder konturer der ligner bolde.
    Returnerer en liste af dicts med center, radius osv.
    """
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

        # radius-filter til jeres nuværende billede
        if radius < 4 or radius > 20:
            continue

        balls.append({
            "center": (int(round(x)), int(round(y))),
            "radius": int(round(radius)),
            "area": float(area),
            "circularity": float(circ),
            "contour": cnt
        })

    return balls
