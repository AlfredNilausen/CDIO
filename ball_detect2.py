import cv2
import numpy as np

# Change this if needed
CAMERA_INDEX = 1

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White detection
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Orange detection
    lower_orange = np.array([10, 120, 120])
    upper_orange = np.array([30, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # Remove noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)

    # Blur helps smooth reflections
    mask = cv2.GaussianBlur(mask, (9,9), 0)
    mask_orange = cv2.GaussianBlur(mask_orange, (9,9), 0)

    contours_white, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours_white:

        area = cv2.contourArea(c)

        if area < 150:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        # Circularity test
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.7:

            (x, y), radius = cv2.minEnclosingCircle(c)

            center = (int(x), int(y))
            radius = int(radius)

            # Draw ball
            cv2.circle(frame, center, radius, (0,255,0), 2)
            cv2.circle(frame, center, 4, (255,0,0), -1)

            text = f"Ball {center}"
            cv2.putText(frame, text, (center[0]+10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    for c in contours_orange:
        area = cv2.contourArea(c)
        if area < 150:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.7:
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))

            cv2.circle(frame, center, int(radius), (0,165,255), 3)  # orange color
            cv2.putText(frame, f"Orange {center}", (center[0]+10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)

    cv2.imshow("mask", mask)
    cv2.imshow("orange mask", mask_orange)
    cv2.imshow("camera", frame)
    

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()