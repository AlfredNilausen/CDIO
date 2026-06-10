#!/usr/bin/env python3
"""
robot_ev3.py  —  kører PÅ EV3'en

Modtager waypoints fra PC via TCP socket og kører motorerne.

Start på EV3:
    python3 robot_ev3.py

Kræver ev3dev + ev3dev-python:
    pip3 install ev3dev2
"""

import socket
import json
import math
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, SpeedPercent, MoveSteering
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_1

# ─────────────────────────────────────────────
# KONFIGURATION  — tilpas til din robot
# ─────────────────────────────────────────────

HOST        = "0.0.0.0"   # lyt på alle interfaces
PORT        = 9999

WHEEL_BASE_MM    = 120    # afstand mellem de to hjul (mm)
WHEEL_DIAM_MM    = 56     # hjuldiameter (mm)

DRIVE_SPEED      = 30     # % af max hastighed ved kørsel
TURN_SPEED       = 20     # % ved drejning
POSITION_TOL_MM  = 30     # acceptabel afstand til waypoint (mm)
ANGLE_TOL_DEG    = 5      # acceptabel vinkelafvigelse (grader)

# ─────────────────────────────────────────────
# MOTORER + SENSOR
# ─────────────────────────────────────────────

steer   = MoveSteering(OUTPUT_A, OUTPUT_B)
gyro    = GyroSensor(INPUT_1)
gyro.mode = "GYRO-ANG"

def reset_gyro():
    gyro.mode = "GYRO-RATE"
    gyro.mode = "GYRO-ANG"

# ─────────────────────────────────────────────
# BEVÆGELSE
# ─────────────────────────────────────────────

def mm_to_degrees(mm):
    """Omregn mm til hjulgrader."""
    circumference = math.pi * WHEEL_DIAM_MM
    return (mm / circumference) * 360

def turn_to_heading(target_heading_deg):
    """Drej på stedet til en given kompas-retning (0=højre, 90=op)."""
    current = gyro.angle
    # Konverter fra vores kompas (0=højre, CCW positiv)
    # til gyro (0=start, CW positiv)
    # Gyro-vinkel stiger ved højresving
    diff = target_heading_deg - current
    # Normaliser til [-180, 180]
    diff = (diff + 180) % 360 - 180

    if abs(diff) < ANGLE_TOL_DEG:
        return

    # Positiv diff = sving mod uret (venstre) i vores system
    steering = -100 if diff > 0 else 100
    rotations = abs(diff) / 360 * (math.pi * WHEEL_BASE_MM / (WHEEL_DIAM_MM * math.pi))
    steer.on_for_rotations(steering, SpeedPercent(TURN_SPEED), rotations)

def drive_straight_mm(distance_mm):
    """Kør ligeud distance_mm (negativ = bakke)."""
    deg = mm_to_degrees(abs(distance_mm))
    rotations = deg / 360
    direction = 1 if distance_mm >= 0 else -1
    steer.on_for_rotations(0, SpeedPercent(DRIVE_SPEED * direction), rotations)

def go_to_waypoint(robot_pos_mm, robot_heading_deg, target_mm):
    """
    Drej mod target og kør dertil.
    robot_pos_mm     : (x, y) aktuel position i mm
    robot_heading_deg: aktuel retning (0=højre, 90=op)
    target_mm        : (x, y) destination i mm
    """
    dx = target_mm[0] - robot_pos_mm[0]
    dy = target_mm[1] - robot_pos_mm[1]
    dist = math.hypot(dx, dy)

    if dist < POSITION_TOL_MM:
        return   # allerede fremme

    # Ønsket retning (0=højre, 90=op — samme som heading-systemet)
    desired_heading = math.degrees(math.atan2(dy, dx))

    turn_to_heading(desired_heading)
    drive_straight_mm(dist)

# ─────────────────────────────────────────────
# KOMMANDO-HÅNDTERING
# ─────────────────────────────────────────────

def handle_command(cmd: dict):
    """
    Kommandoformat fra PC:
      {"type": "waypoints",
       "waypoints": [[x1,y1], [x2,y2], ...],
       "robot_pos": [x, y],
       "heading": 45.0}

      {"type": "stop"}
      {"type": "ping"}
    """
    t = cmd.get("type")

    if t == "ping":
        return {"status": "pong"}

    if t == "stop":
        steer.off()
        return {"status": "stopped"}

    if t == "waypoints":
        waypoints   = cmd["waypoints"]
        robot_pos   = tuple(cmd["robot_pos"])
        heading     = cmd.get("heading", 0.0)
        current_pos = robot_pos

        reset_gyro()

        for wp in waypoints:
            target = tuple(wp)
            print(f"  → waypoint {target}")
            go_to_waypoint(current_pos, heading, target)
            current_pos = target
            # Send acknowledgement after each waypoint so PC can update vision
            # (socket stays open — PC can send next batch after each ACK)

        return {"status": "done", "final_pos": list(current_pos)}

    return {"status": "unknown_command"}

# ─────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────

def main():
    print(f"EV3 robot server lytter på port {PORT}...")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)

    while True:
        conn, addr = srv.accept()
        print(f"Forbundet: {addr}")
        try:
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                # Kommandoer er newline-separerede JSON
                while b"\n" in data:
                    line, data = data.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        cmd  = json.loads(line.decode())
                        resp = handle_command(cmd)
                        conn.sendall((json.dumps(resp) + "\n").encode())
                    except Exception as e:
                        print(f"Fejl: {e}")
                        conn.sendall((json.dumps({"status":"error","msg":str(e)})+"\n").encode())
        finally:
            conn.close()
            print("Forbindelse lukket")

if __name__ == "__main__":
    main()