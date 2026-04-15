import cv2
import numpy as np

# ==================== KONSTANTER ====================
BOARD_WIDTH_MM = 1680
BOARD_HEIGHT_MM = 1235

CAMERA_INDEX = 0

# ==================== HJÆLPEFUNKTIONER ====================

def red_mask(hsv):
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 70, 50])
    upper2 = np.array([180, 255, 255])

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(m1, m2)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(mask, kernel)

def line_angle(x1,y1,x2,y2):
    return abs(np.degrees(np.arctan2(y2-y1, x2-x1)))

def line_length(x1,y1,x2,y2):
    return np.hypot(x2-x1, y2-y1)

def intersect(l1, l2):
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2

    A = np.array([[x2-x1, x3-x4],
                  [y2-y1, y3-y4]])
    B = np.array([x3-x1, y3-y1])

    t, _ = np.linalg.lstsq(A, B, rcond=None)[0]

    x = x1 + t*(x2-x1)
    y = y1 + t*(y2-y1)

    return int(x), int(y)

def find_frame_lines(mask, shape):
    h, w = shape[:2]

    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 120, minLineLength=100, maxLineGap=30)

    if lines is None:
        return None

    best = {"top":None,"bottom":None,"left":None,"right":None}
    best_len = {k:0 for k in best}

    for l in lines:
        x1,y1,x2,y2 = l[0]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        ang = line_angle(x1,y1,x2,y2)
        ln  = line_length(x1,y1,x2,y2)

        if cy < 100 and ang < 25 and ln > best_len["top"]:
            best["top"], best_len["top"] = (x1,y1,x2,y2), ln
        if cy > h-100 and ang < 25 and ln > best_len["bottom"]:
            best["bottom"], best_len["bottom"] = (x1,y1,x2,y2), ln
        if cx < 300 and abs(ang-90) < 25 and ln > best_len["left"]:
            best["left"], best_len["left"] = (x1,y1,x2,y2), ln
        if cx > w-300 and abs(ang-90) < 25 and ln > best_len["right"]:
            best["right"], best_len["right"] = (x1,y1,x2,y2), ln

    if None in best.values():
        return None

    return best

def draw_grid(img):
    for x in range(0, BOARD_WIDTH_MM+1, 200):
        cv2.line(img, (x,0),(x,BOARD_HEIGHT_MM),(50,50,50),1)
        cv2.putText(img, f"{x}", (x+5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(200,200,200),1)

    for y in range(0, BOARD_HEIGHT_MM+1, 200):
        cv2.line(img, (0,y),(BOARD_WIDTH_MM,y),(50,50,50),1)
        cv2.putText(img, f"{y}", (5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(200,200,200),1)

# ==================== MAIN ====================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)

    lines = find_frame_lines(mask, frame.shape)
    if lines is None:
        cv2.imshow("camera", frame)
        if cv2.waitKey(1)==27: break
        continue

    tl = intersect(lines["top"], lines["left"])
    tr = intersect(lines["top"], lines["right"])
    bl = intersect(lines["bottom"], lines["left"])
    br = intersect(lines["bottom"], lines["right"])

    for p in [tl,tr,bl,br]:
        cv2.circle(frame, p, 8, (0,255,0), -1)

    src = np.array([bl, tl, tr, br], dtype="float32")
    dst = np.array([[0,0],[0,BOARD_HEIGHT_MM],
                    [BOARD_WIDTH_MM,BOARD_HEIGHT_MM],
                    [BOARD_WIDTH_MM,0]], dtype="float32")

    H = cv2.getPerspectiveTransform(src, dst)
    world = cv2.warpPerspective(frame, H, (BOARD_WIDTH_MM, BOARD_HEIGHT_MM))

    draw_grid(world)

    cv2.imshow("camera", frame)
    cv2.imshow("world (mm)", world)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()