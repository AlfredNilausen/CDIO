import cv2
import numpy as np

def get_mask(hsv, lower, upper, do_morph=True):
    mask = cv2.inRange(hsv, lower, upper)

    if do_morph:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    return mask


def find_contours(mask, min_area=100, mode=cv2.RETR_EXTERNAL):
    contours, _ = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]


def is_circle(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        return False

    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return circularity > 0.7


CAMERA_INDEX = 1

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame")
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

    # gentler than strong dilation
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # border contours only
    red_external = find_contours(mask_red, min_area=500, mode=cv2.RETR_EXTERNAL)

    largest_red = None
    roi_mask = np.zeros(mask_red.shape, dtype=np.uint8)
    mask_red_inside = np.zeros(mask_red.shape, dtype=np.uint8)

    if red_external:
        largest_red = max(red_external, key=cv2.contourArea)
        cv2.drawContours(frame, [largest_red], -1, (0, 0, 255), 3)

        # fill border contour to get field region
        cv2.drawContours(roi_mask, [largest_red], -1, 255, thickness=cv2.FILLED)

        # shrink so the border itself is mostly excluded
        roi_mask = cv2.erode(roi_mask, np.ones((15, 15), np.uint8), iterations=1)

        # only red inside the field
        mask_red_inside = cv2.bitwise_and(mask_red, roi_mask)

        # now search all contours inside the field
        inside_contours = find_contours(mask_red_inside, min_area=200, mode=cv2.RETR_LIST)

        for cnt in inside_contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # roughly square-ish and not circular => possible cross
            if 0.6 < aspect_ratio < 1.4 and 500 < area < 8000 and circularity < 0.75:
                cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 3)
                cv2.putText(frame, "Cross", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # ---------------- WHITE BALLS ----------------
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])

    mask_white = get_mask(hsv, lower_white, upper_white)
    if largest_red is not None:
        mask_white = cv2.bitwise_and(mask_white, roi_mask)

    white_contours = find_contours(mask_white, min_area=50, mode=cv2.RETR_EXTERNAL)

    for cnt in white_contours:
        if is_circle(cnt):
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius > 5:
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 255, 255), 2)

    # ---------------- ORANGE BALLS ----------------
    lower_orange = np.array([5, 120, 120])
    upper_orange = np.array([25, 255, 255])

    mask_orange = get_mask(hsv, lower_orange, upper_orange)
    if largest_red is not None:
        mask_orange = cv2.bitwise_and(mask_orange, roi_mask)

    orange_contours = find_contours(mask_orange, min_area=50, mode=cv2.RETR_EXTERNAL)

    for cnt in orange_contours:
        if is_circle(cnt):
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius > 5:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 140, 255), 2)

    # ---------------- DISPLAY ----------------
    cv2.imshow("camera", frame)
    cv2.imshow("red mask", mask_red)
    cv2.imshow("roi mask", roi_mask)
    cv2.imshow("red inside roi", mask_red_inside)
    cv2.imshow("white mask", mask_white)
    cv2.imshow("orange mask", mask_orange)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()