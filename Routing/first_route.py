import cv2
import numpy as np
import math
import os

# ---------------------------
# Setup
# ---------------------------

BASE_DIR = os.path.dirname(__file__)
IMAGE_PATH = os.path.join(BASE_DIR, "data", "image.png")

BALL_RADIUS = 8

# ---------------------------
# Modes
# ---------------------------

MODE_NONE = 0
MODE_ADD_WHITE = 1
MODE_ADD_ORANGE = 2
MODE_MOVE_START = 3
MODE_MOVE_CENTER = 4
MODE_MOVE_OBJECT = 5

mode = MODE_NONE

# ---------------------------
# State
# ---------------------------

white_balls = []
orange_balls = []

start_position = (50, 50)
center = (300, 300)

selected = None
dragging = False

# ---------------------------
# Math
# ---------------------------

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_quadrant(p):
    x, y = p[0] - center[0], p[1] - center[1]

    if x < 0 and y < 0:
        return 1
    elif x > 0 and y < 0:
        return 2
    elif x > 0 and y > 0:
        return 3
    elif x < 0 and y > 0:
        return 4
    return 0

def nearest_neighbor(start, pts):
    path = []
    current = start
    pts = pts.copy()

    while pts:
        nxt = min(pts, key=lambda p: distance(current, p))
        path.append(nxt)
        current = nxt
        pts.remove(nxt)

    return path

def plan_route():
    balls = white_balls + orange_balls

    quadrants = {1: [], 2: [], 3: [], 4: []}
    for b in balls:
        q = get_quadrant(b)
        if q:
            quadrants[q].append(b)

    offset = 200
    gates = {
        1: (center[0], center[1] - offset),
        2: (center[0] + offset, center[1]),
        3: (center[0], center[1] + offset),
        4: (center[0] - offset, center[1]),
    }

    route = []
    current = start_position

    for q in [1, 2, 3, 4]:
        if not quadrants[q]:
            continue

        # 1. collect balls first
        sub = nearest_neighbor(current, quadrants[q])
        route.extend(sub)
        current = sub[-1]

        # 2. then go to that quadrant's gate (exit point)
        gate = gates[q]
        route.append(gate)
        current = gate

    return route

# ---------------------------
# Mouse
# ---------------------------

def mouse(event, x, y, flags, param):
    global dragging, selected, start_position, center

    if event == cv2.EVENT_LBUTTONDOWN:

        if mode == MODE_ADD_WHITE:
            white_balls.append((x, y))

        elif mode == MODE_ADD_ORANGE:
            orange_balls.append((x, y))

        elif mode == MODE_MOVE_START:
            start_position = (x, y)

        elif mode == MODE_MOVE_CENTER:
            center = (x, y)

        elif mode == MODE_MOVE_OBJECT:
            # check all objects
            for i, b in enumerate(white_balls):
                if distance((x, y), b) < BALL_RADIUS * 2:
                    selected = ("white", i)
                    dragging = True
                    return

            for i, b in enumerate(orange_balls):
                if distance((x, y), b) < BALL_RADIUS * 2:
                    selected = ("orange", i)
                    dragging = True
                    return

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and selected:
            t, i = selected
            if t == "white":
                white_balls[i] = (x, y)
            else:
                orange_balls[i] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        selected = None

# ---------------------------
# Main
# ---------------------------

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise Exception("Image not found")

h, w = img.shape[:2]
center = (w // 2, h // 2)

cv2.namedWindow("Track")
cv2.setMouseCallback("Track", mouse)

route = []

while True:
    display = img.copy()

    # draw cross
    cv2.drawMarker(display, center, (0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2)

    # draw quadrant lines
    cv2.line(display, (center[0], 0), (center[0], h), (255, 0, 0), 1)
    cv2.line(display, (0, center[1]), (w, center[1]), (255, 0, 0), 1)

    # draw balls
    for b in white_balls:
        cv2.circle(display, b, BALL_RADIUS, (255, 255, 255), -1)

    for b in orange_balls:
        cv2.circle(display, b, BALL_RADIUS, (0, 165, 255), -1)

    # draw start
    cv2.circle(display, start_position, BALL_RADIUS, (255, 0, 0), -1)

    # draw route
    if route:
        prev = start_position
        for p in route:
            cv2.line(display, prev, p, (0, 255, 0), 2)
            prev = p

    # UI text
    mode_text = {
        MODE_NONE: "NONE",
        MODE_ADD_WHITE: "ADD WHITE (w)",
        MODE_ADD_ORANGE: "ADD ORANGE (o)",
        MODE_MOVE_START: "MOVE START (s)",
        MODE_MOVE_CENTER: "MOVE CENTER (x)",
        MODE_MOVE_OBJECT: "MOVE OBJECTS (m)"
    }

    cv2.putText(display, f"Mode: {mode_text[mode]}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Track", display)

    key = cv2.waitKey(30)

    if key == 27:
        break
    elif key == ord('w'):
        mode = MODE_ADD_WHITE
    elif key == ord('o'):
        mode = MODE_ADD_ORANGE
    elif key == ord('s'):
        mode = MODE_MOVE_START
    elif key == ord('x'):
        mode = MODE_MOVE_CENTER
    elif key == ord('m'):
        mode = MODE_MOVE_OBJECT
    elif key == ord('r'):
        route = plan_route()
        print("Route:", route)
    elif key == ord('c'):
        white_balls.clear()
        orange_balls.clear()
        route = []

cv2.destroyAllWindows()