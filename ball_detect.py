import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White color range (ping pong balls)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Clean up noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:

        area = cv2.contourArea(c)

        # Filter very small objects
        if area > 200:

            (x, y), radius = cv2.minEnclosingCircle(c)

            center = (int(x), int(y))
            radius = int(radius)

            # Draw circle around detected ball
            cv2.circle(frame, center, radius, (0,255,0), 2)

            # Draw center point
            cv2.circle(frame, center, 5, (255,0,0), -1)

            # Print coordinates
            text = f"Ball: {center}"
            cv2.putText(frame, text, (center[0]+10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Show mask and camera
    cv2.imshow("mask", mask)
    cv2.imshow("camera", frame)

    # Press ESC to quit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()