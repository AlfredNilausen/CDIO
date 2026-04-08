import cv2
import numpy as np

CAMERA_INDEX = 1

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    frame = cv2.resize(frame, (640, 480))

    # =========================
# 1. DETECT RED BORDER
# =========================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Expanded red range to capture darker/light reds
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

# Dilate to connect broken pieces, then clean up
    kernel = np.ones((7,7), np.uint8)
    mask_red = cv2.dilate(mask_red, kernel, iterations=2)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

# Find contours and approximate largest rectangle
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect = None
    best_area = 0

# Find the largest contour in the red mask
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

    # Minimum area rectangle (tight fit)
        rect = cv2.minAreaRect(largest_contour)  # ((cx,cy), (w,h), angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)  # convert to integer coordinates

    # Optional: shrink slightly to avoid touching tape
        shrink_pixels = 5
    # Move each corner slightly towards the center
        center = np.mean(box, axis=0)
        box = np.array([((p - center) * 0.95 + center) for p in box], dtype=np.int0)

    # Draw the rectangle on the camera feed
        cv2.polylines(frame, [box], isClosed=True, color=(0,255,255), thickness=3)

    # Fill mask_border for excluding balls
        mask_border = np.zeros_like(mask_red)
        cv2.drawContours(mask_border, [box], -1, 255, -1)

# Show mask
    cv2.imshow("red mask", mask_red)
    cv2.imshow("mask border", mask_border)

    # =========================
    # 2. BALL DETECTION
    # =========================
    # White balls
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Orange ball
    lower_orange = np.array([10, 120, 120])
    upper_orange = np.array([30, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # Remove border area
    mask_white = cv2.bitwise_and(mask_white, cv2.bitwise_not(mask_border))
    mask_orange = cv2.bitwise_and(mask_orange, cv2.bitwise_not(mask_border))

    # Clean masks
    kernel = np.ones((5,5), np.uint8)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)

    # =========================
    # 3. FIND BALLS
    # =========================
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ---- WHITE BALLS ----
    for c in contours_white:
        area = cv2.contourArea(c)
        if area < 200 or area > 2000:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.7:
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            cv2.circle(frame, center, int(radius), (0,255,0), 2)
            cv2.circle(frame, center, 4, (255,0,0), -1)
            cv2.putText(frame, f"White {center}", (center[0]+10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # ---- ORANGE BALL ----
    for c in contours_orange:
        area = cv2.contourArea(c)
        if area < 200 or area > 2000:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.7:
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            cv2.circle(frame, center, int(radius), (0,165,255), 3)
            cv2.circle(frame, center, 4, (0,0,255), -1)
            cv2.putText(frame, f"Orange {center}", (center[0]+10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)

    # =========================
    # 4. SHOW WINDOWS
    # =========================
    cv2.imshow("camera", frame)
    cv2.imshow("mask border", mask_border)
    cv2.imshow("red mask", mask_red)
    cv2.imshow("white mask", mask_white)
    cv2.imshow("orange mask", mask_orange)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()