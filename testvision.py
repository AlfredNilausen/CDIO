import cv2
import numpy as np

def get_mask(hsv, lower, upper, do_morph=True):
    mask = cv2.inRange(hsv, lower, upper)

    if do_morph:
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    return mask


def find_contours(mask, min_area=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]


def is_circle(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        return False

    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return circularity > 0.7


CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ---------------- RED MASK ----------------
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_red = cv2.bitwise_or(
        get_mask(hsv, lower_red1, upper_red1, False),
        get_mask(hsv, lower_red2, upper_red2, False)
    )

    # 🔧 connect cross parts
    kernel = np.ones((5,5), np.uint8)
    mask_red = cv2.dilate(mask_red, kernel)

    red_contours = find_contours(mask_red, min_area=300)

    # -------- Find largest contour (border) --------
    largest_red = None
    max_area = 0

    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_red = cnt

    # Draw border
    if largest_red is not None:
        cv2.drawContours(frame, [largest_red], -1, (0, 0, 255), 3)

    # -------- CROSS detection (NEW METHOD) --------
    for cnt in red_contours:
        if cnt is largest_red:
            continue

        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        if 0.5 < aspect_ratio < 1.5 and 200 < area < 5000:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 3)

    # ---------------- WHITE BALLS ----------------
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])

    mask_white = get_mask(hsv, lower_white, upper_white)
    white_contours = find_contours(mask_white, min_area=50)

    for cnt in white_contours:
        if is_circle(cnt):
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius > 5:
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 255, 255), 2)

    # ---------------- ORANGE BALL ----------------
    lower_orange = np.array([5, 120, 120])
    upper_orange = np.array([25, 255, 255])

    mask_orange = get_mask(hsv, lower_orange, upper_orange)
    orange_contours = find_contours(mask_orange, min_area=50)

    for cnt in orange_contours:
        if is_circle(cnt):
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius > 5:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 140, 255), 2)

    # ---------------- DISPLAY ----------------
    cv2.imshow('camera', frame)
    cv2.imshow('red mask', mask_red)
    cv2.imshow('white mask', mask_white)
    cv2.imshow('orange mask', mask_orange)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()