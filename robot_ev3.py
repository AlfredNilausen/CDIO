#!/usr/bin/env python3
"""
robot_ev3.py  -  korer PA EV3

Motor A = hojre forhjul
Motor D = venstre forhjul
Motor C = opsamlingsmekanisme
Ingen gyro - drejer via hjulomkreds-beregning
"""

import socket
import json
import math
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_C, OUTPUT_D, SpeedPercent

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────

HOST = "0.0.0.0"
PORT = 9999

WHEEL_BASE_MM   = 750   # afstand mellem hjulene (maal efter paa robotten)
WHEEL_DIAM_MM   = 65    # hjuldiameter (maal efter paa robotten)

DRIVE_SPEED     = 30    # % ved korslen
TURN_SPEED      = 20    # % ved drejning
COLLECT_SPEED   = 50    # % ved opsamling
COLLECT_TIME_S  = 1.5   # sekunder opsamlingsmekanisme koerer
POSITION_TOL_MM = 40    # acceptabel afstand til waypoint

# ─────────────────────────────────────────────
# MOTORER
# ─────────────────────────────────────────────

motor_right  = LargeMotor(OUTPUT_A)
motor_left   = LargeMotor(OUTPUT_D)
motor_collect = LargeMotor(OUTPUT_C)
motor_right.polarity = "inversed"
motor_left.polarity  = "inversed"

def motors_stop():
    motor_right.off()
    motor_left.off()

def mm_to_rotations(mm):
    return mm / (math.pi * WHEEL_DIAM_MM)

def degrees_to_rotations(angle_deg):
    """Rotationer per hjul for at dreje angle_deg grader paa stedet."""
    arc = (abs(angle_deg) / 360.0) * math.pi * WHEEL_BASE_MM
    return arc / (math.pi * WHEEL_DIAM_MM)

def drive_straight(distance_mm):
    """Koer ligeud (negativ = bakke)."""
    rot = mm_to_rotations(abs(distance_mm))
    sp  = DRIVE_SPEED if distance_mm >= 0 else -DRIVE_SPEED
    motor_right.on_for_rotations(SpeedPercent(sp),  rot, block=False)
    motor_left.on_for_rotations( SpeedPercent(sp),  rot, block=True)
    motors_stop()

def turn_degrees(angle_deg):
    """
    Drej paa stedet.
    Positiv = mod uret (venstre), negativ = med uret (hojre).
    """
    if abs(angle_deg) < 3:
        return
    rot = degrees_to_rotations(angle_deg)
    sp  = TURN_SPEED
    if angle_deg > 0:   # venstre
        motor_right.on_for_rotations(SpeedPercent( sp), rot, block=False)
        motor_left.on_for_rotations( SpeedPercent(-sp), rot, block=True)
    else:               # hojre
        motor_right.on_for_rotations(SpeedPercent(-sp), rot, block=False)
        motor_left.on_for_rotations( SpeedPercent( sp), rot, block=True)
    motors_stop()

def collect():
    """Koer opsamlingsmekanismen i COLLECT_TIME_S sekunder."""
    import time
    motor_collect.on(SpeedPercent(COLLECT_SPEED))
    time.sleep(COLLECT_TIME_S)
    motor_collect.off()

# ─────────────────────────────────────────────
# NAVIGATION  (uden gyro - bruger heading fra vision)
# ─────────────────────────────────────────────

current_heading = 0.0   # holdes ajour lokalt mellem waypoints

def go_to_waypoint(robot_pos, heading_deg, target):
    """
    Beregn retning og afstand til target, drej og koer.
    robot_pos / target : (x_mm, y_mm)
    heading_deg        : nuvaerende retning (0=hoejre, 90=op)
    Returnerer ny heading.
    """
    global current_heading
    dx = target[0] - robot_pos[0]
    dy = target[1] - robot_pos[1]
    dist = math.hypot(dx, dy)

    if dist < POSITION_TOL_MM:
        return heading_deg

    desired = math.degrees(math.atan2(dy, dx))
    diff    = desired - heading_deg
    # Normaliser til [-180, 180]
    diff = (diff + 180) % 360 - 180

    turn_degrees(diff)
    drive_straight(dist)

    return desired

# ─────────────────────────────────────────────
# KOMMANDO-HAANDTERING
# ─────────────────────────────────────────────

def handle_command(cmd):
    t = cmd.get("type")

    if t == "ping":
        return {"status": "pong"}

    if t == "stop":
        motors_stop()
        return {"status": "stopped"}

    if t == "collect":
        collect()
        return {"status": "collected"}

    if t == "waypoints":
        waypoints   = cmd["waypoints"]
        robot_pos   = tuple(cmd["robot_pos"])
        heading     = cmd.get("heading", 0.0)
        do_collect  = cmd.get("collect_at_each", False)
        current_pos = robot_pos

        for i, wp in enumerate(waypoints):
            target = tuple(wp)
            print("  -> waypoint {} af {} : {}".format(i+1, len(waypoints), target))
            heading = go_to_waypoint(current_pos, heading, target)
            current_pos = target

            # Opsaml hvis waypoint er markeret som bold-position
            if do_collect and cmd.get("collect_indices") and i in cmd["collect_indices"]:
                collect()

        return {"status": "done", "final_pos": list(current_pos), "heading": heading}

    return {"status": "unknown"}

# ─────────────────────────────────────────────
# TCP SERVER
# ─────────────────────────────────────────────

def main():
    print("EV3 server klar paa port {}".format(PORT))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)

    while True:
        conn, addr = srv.accept()
        print("Forbundet fra {}".format(addr))
        buf = b""
        try:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        cmd  = json.loads(line.decode())
                        resp = handle_command(cmd)
                        conn.sendall((json.dumps(resp) + "\n").encode())
                    except Exception as e:
                        print("Fejl: {}".format(e))
                        conn.sendall((json.dumps({"status":"error","msg":str(e)})+"\n").encode())
        finally:
            conn.close()
            print("Forbindelse lukket")

if __name__ == "__main__":
    main()