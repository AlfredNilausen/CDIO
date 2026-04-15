import cv2
import numpy as np

from red_mask import red_mask


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


def make_cross_search_mask(frame_shape, corners, margin_px=35):
    """
    Søgemaske til korset:
    - starter fra hele banen
    - eroderer banen indad, så den røde ramme forsvinder
    """
    board_mask = make_board_mask(frame_shape, corners)

    ksize = 2 * margin_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    inner_mask = cv2.erode(board_mask, kernel, iterations=1)

    return inner_mask


def get_red_cross_mask(frame, corners, margin_px=35):
    """
    Finder rød maske inde i den centrale del af banen.
    Returnerer:
    - cross_mask
    - search_mask
    - full_red_mask
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    full_red_mask = red_mask(hsv)

    search_mask = make_cross_search_mask(frame.shape, corners, margin_px=margin_px)
    cross_mask = cv2.bitwise_and(full_red_mask, search_mask)

    # Rens lidt op
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cross_mask = cv2.morphologyEx(cross_mask, cv2.MORPH_OPEN, kernel)
    cross_mask = cv2.morphologyEx(cross_mask, cv2.MORPH_CLOSE, kernel)

    return cross_mask, search_mask, full_red_mask


def contour_center(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def find_cross_candidate(cross_mask, corners):
    """
    Finder bedste kontur for det røde kryds.
    Kriterier:
    - passende areal
    - nogenlunde kvadratisk bounding box
    - konturen ligger nær banens centrum
    - extent passer bedre til kryds end til en fuld firkant
    """
    contours, _ = cv2.findContours(cross_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    board_center_x = int((corners["TL"][0] + corners["TR"][0] + corners["BL"][0] + corners["BR"][0]) / 4)
    board_center_y = int((corners["TL"][1] + corners["TR"][1] + corners["BL"][1] + corners["BR"][1]) / 4)

    best = None
    best_score = -1e18

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150 or area > 20000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue

        aspect_ratio = w / float(h)
        if not (0.6 <= aspect_ratio <= 1.4):
            continue

        extent = area / float(w * h)
        # Kryds fylder typisk mindre end en fuld firkant i sin bounding box
        if not (0.15 <= extent <= 0.55):
            continue

        center = contour_center(cnt)
        if center is None:
            continue

        dist = np.hypot(center[0] - board_center_x, center[1] - board_center_y)

        # Score:
        # - større areal er godt
        # - tæt på midten er godt
        # - extent omkring ~0.30-0.40 er ofte fint for et kryds
        extent_penalty = abs(extent - 0.32) * 500.0
        score = area - 4.0 * dist - extent_penalty

        if score > best_score:
            best_score = score
            best = {
                "contour": cnt,
                "center": center,
                "bbox": (x, y, w, h),
                "area": area,
                "aspect_ratio": aspect_ratio,
                "extent": extent,
                "distance_to_center": dist,
                "score": score
            }

    return best