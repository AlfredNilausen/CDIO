"""
test_turn_to_ball.py  -  korer pa PC

Kameraet ser robotten (ArUco) og en bold.
Tryk ENTER pa EV3 -> robotten drejer mod bolden og stopper.
Flyt bolden og tryk ENTER igen for at gentage.

Krav:
  - EV3 korer turn_server_ev3.py  (port 9998)
  - ArUco markør ID 0 er synlig fra kameraet
  - En hvid eller orange bold er synlig

Konfiguration:
  EV3_HOST, CAMERA_INDEX, HEADING_OFFSET, TURN_TOLERANCE
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../vision_bestfit'))

import cv2
import cv2.aruco as aruco
import numpy as np
import socket, json, math, time

# ── Konfiguration ─────────────────────────────────────────────────────────────
EV3_HOST       = "192.168.0.1"
EV3_PORT       = 9998
CAMERA_INDEX   = 1
MARKER_ID      = 0
MARKER_SIZE_MM = 80
HEADING_OFFSET = -88     # juster i trin af 90 hvis pilen peger forkert
TURN_TOLERANCE = 1.0     # grader -- stopper indenfor dette af maalet
OVERSHOOT_COMP = 12.0     # grader -- stop lidt foer maalet for at kompensere for glid

# Bold-detektion
WHITE_S_MAX   = 130
WHITE_V_MIN   = 120
ORANGE_H_MIN  = 12
ORANGE_H_MAX  = 35
ORANGE_S_MIN  = 70
HOUGH_PARAM2  = 18
MIN_RADIUS    = 6
MAX_RADIUS    = 14
# ─────────────────────────────────────────────────────────────────────────────

# ── ArUco ─────────────────────────────────────────────────────────────────────
_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
_params = aruco.DetectorParameters()
_det    = aruco.ArucoDetector(_dict, _params)
_half   = MARKER_SIZE_MM / 2.0
_OBJ    = np.array([[-_half,_half,0],[_half,_half,0],
                    [ _half,-_half,0],[-_half,-_half,0]], dtype=np.float32)

def get_robot(frame):
    """Returnerer (heading_deg, center_px) eller (None, None)."""
    H, W  = frame.shape[:2]
    f     = max(W, H)
    cam   = np.array([[f,0,W/2],[0,f,H/2],[0,0,1]], dtype=np.float64)
    dist  = np.zeros((5,1), dtype=np.float64)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _det.detectMarkers(gray)
    if ids is None:
        return None, None
    for i, mid in enumerate(ids.flatten()):
        if mid != MARKER_ID: continue
        mc = corners[i][0]
        ok, rvec, _ = cv2.solvePnP(_OBJ, mc.astype(np.float32),
                                    cam, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok: continue
        R, _ = cv2.Rodrigues(rvec)
        heading = math.degrees(math.atan2(-R[1,0], R[0,0])) + HEADING_OFFSET
        center  = (int(mc[:,0].mean()), int(mc[:,1].mean()))
        return heading, center
    return None, None

def get_ball(frame):
    """
    Returnerer center (x,y) af den tydeligste bold (hvid eller orange).
    Vælger den bold der er tættest på billedets centrum.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15,15), 1)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H_ch, S, V = cv2.split(hsv)

    white_mask  = ((S < WHITE_S_MAX) & (V > WHITE_V_MIN)).astype(np.uint8)*255
    orange_mask = ((H_ch >= ORANGE_H_MIN) & (H_ch <= ORANGE_H_MAX) &
                   (S >= ORANGE_S_MIN)).astype(np.uint8)*255

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=20, param1=200, param2=HOUGH_PARAM2,
        minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)

    if circles is None:
        return None

    h, w = frame.shape[:2]
    best = None
    best_score = -1

    for (cx, cy, r) in np.round(circles[0]).astype(int):
        x0,x1 = max(0,cx-r), min(w,cx+r)
        y0,y1 = max(0,cy-r), min(h,cy+r)
        if x1<=x0 or y1<=y0: continue
        wo = np.mean(white_mask [y0:y1,x0:x1] > 0)
        oo = np.mean(orange_mask[y0:y1,x0:x1] > 0)
        score = max(wo, oo)
        if score > 0.25 and score > best_score:
            best_score = score
            best = (cx, cy)

    return best

# ── EV3 socket ────────────────────────────────────────────────────────────────
sock = None

def ev3_connect():
    global sock
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect((EV3_HOST, EV3_PORT))
        print("[EV3] Forbundet til {}:{}".format(EV3_HOST, EV3_PORT))
        return True
    except Exception as e:
        print("[EV3] Forbindelsesfejl: {} -- korer uden robot".format(e))
        return False

def ev3_send(cmd):
    if sock is None: return None
    try:
        sock.sendall((json.dumps(cmd)+"\n").encode())
        buf = b""
        while b"\n" not in buf:
            buf += sock.recv(512)
        return json.loads(buf.split(b"\n")[0].decode())
    except:
        return None

def ev3_poll_button():
    """Returnerer True hvis ENTER er trykket pa EV3 siden sidst."""
    r = ev3_send({"type": "wait_ready"})
    return r is not None and r.get("status") == "ready"

def ev3_show(line1, line2=""):
    ev3_send({"type": "show", "line1": line1, "line2": line2})

# ── Drej mod bold ─────────────────────────────────────────────────────────────
def angle_diff(target, current):
    return (target - current + 180) % 360 - 180

def turn_to_ball(cap, ball_px, robot_center, robot_heading):
    """
    Beregner vinkel fra robot til bold,
    starter drejning og stopper nar heading matcher.
    """
    # Beregn onsket heading (kamera-koordinater: y er nedad)
    dx = ball_px[0] - robot_center[0]
    dy = ball_px[1] - robot_center[1]
    # I kamera: positiv y er ned. I heading-system: 0=hoejre, 90=op (negativ y)
    target_heading = math.degrees(math.atan2(-dy, dx))

    diff = angle_diff(target_heading, robot_heading)
    print("  Robot heading: {:.1f}  ->  Bold heading: {:.1f}  (drej {:.1f})".format(
          robot_heading, target_heading, diff))

    if abs(diff) < TURN_TOLERANCE:
        print("  Allerede rettet mod bolden!")
        return True

    # Kompenser for overshooting
    stop_target = target_heading - (OVERSHOOT_COMP if diff > 0 else -OVERSHOOT_COMP)

    # Start drejning i rigtig retning
    direction = "turn_left" if diff > 0 else "turn_right"
    ev3_send({"type": direction})
    ev3_show("Drejer...", "{:.0f} grader".format(abs(diff)))

    start_time = time.time()
    history = []

    while True:
        ret, frame = cap.read()
        if not ret: continue

        h, center = get_robot(frame)
        if h is not None:
            history.append(h)
            if len(history) > 4:
                h = sum(history[-4:]) / 4   # udglaet heading

            remaining = angle_diff(target_heading, h)

            # Stop-check
            if abs(angle_diff(stop_target, h)) <= TURN_TOLERANCE:
                ev3_send({"type": "stop"})
                time.sleep(0.1)
                # Maaling efter stop (vent pa robotten er stille)
                time.sleep(0.3)
                ret2, frame2 = cap.read()
                h_final, _ = get_robot(frame2 if ret2 else frame)
                if h_final is None: h_final = h
                err = abs(angle_diff(target_heading, h_final))
                ok  = err <= TURN_TOLERANCE * 2
                print("  Stoppet ved {:.1f} deg  |  Fejl: {:.1f} deg  |  {}".format(
                      h_final, err, "PASS" if ok else "FAIL"))
                return ok

        if time.time() - start_time > 8:
            ev3_send({"type": "stop"})
            print("  [TIMEOUT]")
            return False

        yield frame, h, target_heading   # yield til draw-loop

    return False

# ── Tegn overlay ──────────────────────────────────────────────────────────────
def draw_overlay(frame, robot_h, robot_center, ball_center, target_h, state, results):
    disp = frame.copy()
    H, W = disp.shape[:2]

    # Bolt
    if ball_center:
        cv2.circle(disp, ball_center, 14, (0,200,255), 2)
        cv2.circle(disp, ball_center,  4, (0,200,255), -1)
        cv2.putText(disp, "BOLD", (ball_center[0]+12, ball_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)

    if robot_center and robot_h is not None:
        cx, cy = robot_center

        # Nuvaerende retning (groen)
        ex = int(cx + 80*math.cos(math.radians(robot_h)))
        ey = int(cy - 80*math.sin(math.radians(robot_h)))
        cv2.arrowedLine(disp,(cx,cy),(ex,ey),(0,255,0),3,tipLength=0.25)

        # Maalretning mod bold (gul)
        if target_h is not None:
            ex2 = int(cx + 80*math.cos(math.radians(target_h)))
            ey2 = int(cy - 80*math.sin(math.radians(target_h)))
            cv2.arrowedLine(disp,(cx,cy),(ex2,ey2),(0,220,255),2,tipLength=0.25)

        # Linje fra robot til bold
        if ball_center:
            cv2.line(disp, robot_center, ball_center, (100,100,100), 1)

        cv2.circle(disp, (cx,cy), 8, (255,180,0), -1)

    # Status-boks oppe til venstre
    cv2.rectangle(disp,(0,0),(W,90),(0,0,0),-1)

    h_txt  = "{:.1f} deg".format(robot_h)  if robot_h  is not None else "?"
    th_txt = "{:.1f} deg".format(target_h) if target_h is not None else "?"
    robot_ok = robot_h is not None
    ball_ok  = ball_center is not None

    cv2.putText(disp,"Robot: {}  Bold: {}".format(
        "FUNDET" if robot_ok else "IKKE FUNDET",
        "FUNDET" if ball_ok  else "IKKE FUNDET"),
        (10,22), cv2.FONT_HERSHEY_SIMPLEX,0.6,
        (0,255,0) if (robot_ok and ball_ok) else (0,100,255), 2)
    cv2.putText(disp,"Heading: {}  ->  Maal: {}".format(h_txt, th_txt),
        (10,48), cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,0),1)
    cv2.putText(disp, state,
        (10,72), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),1)

    # Resultater nederst
    y = H - 10
    for label, res in reversed(list(results.items())[-4:]):
        col = (0,255,0) if res=="PASS" else (0,0,255)
        cv2.putText(disp,"{}: {}".format(label,res),(10,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1)
        y -= 20

    cv2.putText(disp,"MELLEMRUM = start  |  q = afslut  |  +/- = tolerance ({:.0f} grader)".format(TURN_TOLERANCE),
        (10,H-90),cv2.FONT_HERSHEY_SIMPLEX,0.4,(120,120,120),1)

    return disp

# ── Main ──────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

connected = ev3_connect()
if connected:
    ev3_show("Klar!", "PC styrer")

results   = {}
test_num  = 0
state     = "Placer bold + robot, tryk ENTER pa EV3"
turning   = False
gen       = None   # turn-generator

print("Klar. Tryk MELLEMRUM for at starte en test. q for at afslutte.")

while True:
    ret, frame = cap.read()
    if not ret: continue

    robot_h, robot_center = get_robot(frame)
    ball_center           = get_ball(frame)

    # Beregn maalheading
    target_h = None
    if robot_center and ball_center:
        dx = ball_center[0] - robot_center[0]
        dy = ball_center[1] - robot_center[1]
        target_h = math.degrees(math.atan2(-dy, dx))

    # Koer drej-generator
    if turning and gen is not None:
        try:
            frame_g, h_g, th_g = next(gen)
            if h_g is not None:   robot_h      = h_g
            if th_g is not None:  target_h     = th_g
            frame = frame_g
        except StopIteration as e:
            ok     = bool(e.value) if e.value is not None else False
            label  = "Test{}".format(test_num)
            results[label] = "PASS" if ok else "FAIL"
            state  = "{} -> {}  |  Flyt bold og tryk MELLEMRUM".format(label, results[label])
            turning = False
            gen     = None
            if connected:
                ev3_show("PASS" if ok else "FAIL", "")
            print(state)

    disp = draw_overlay(frame, robot_h, robot_center, ball_center, target_h, state, results)
    cv2.imshow("Test: Drej mod bold", disp)

    key = cv2.waitKey(15) & 0xFF
    if key == ord('q'): break
    elif key == ord('+') or key == ord('='):
        TURN_TOLERANCE = min(20, TURN_TOLERANCE+1)
        print("Tolerance: {:.0f} grader".format(TURN_TOLERANCE))
    elif key == ord('-'):
        TURN_TOLERANCE = max(1,  TURN_TOLERANCE-1)
        print("Tolerance: {:.0f} grader".format(TURN_TOLERANCE))

    # MELLEMRUM pa PC starter test
    if not turning and key == ord(' '):
        if robot_h is None:
            state = "FEJL: Robot ikke fundet - flyt markoren"
            ev3_show("Ingen robot", "prøv igen")
        elif ball_center is None:
            state = "FEJL: Ingen bold fundet"
            ev3_show("Ingen bold", "prøv igen")
        else:
            test_num += 1
            state    = "Drejer mod bold... (test {})".format(test_num)
            print("\n[TEST {}] Starter".format(test_num))
            turning  = True
            gen      = turn_to_ball(cap, ball_center, robot_center, robot_h)

print("\n=== RESULTATER ===")
for label, res in results.items():
    print("  {:15s}  {}".format(label, res))
total = len(results)
passed = sum(1 for r in results.values() if r=="PASS")
print("  {}/{} PASS".format(passed, total))
print("==================")

if sock: sock.close()
cap.release()
cv2.destroyAllWindows()