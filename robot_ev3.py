#!/usr/bin/env python3
"""
robot_ev3.py - korer PA EV3

Knapper:
  ENTER  = stop / genoptag
  OP     = opsamlingsmotor frem
  NED    = opsamlingsmotor baglens (skub ud)
  VENSTRE/HOEJRE = reserver til fremtidigt brug
"""

import math
import socket
import json
import threading
import time
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_C, OUTPUT_D, SpeedPercent
from ev3dev2.button import Button

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────

HOST            = "0.0.0.0"
PORT            = 9999

WHEEL_BASE_MM   = 120
WHEEL_DIAM_MM   = 56

DRIVE_SPEED     = 30
TURN_SPEED      = 20
COLLECT_SPEED   = 50
COLLECT_TIME_S  = 1.5
POSITION_TOL_MM = 40

# ─────────────────────────────────────────────
# MOTORER + KNAPPER
# ─────────────────────────────────────────────

motor_right   = LargeMotor(OUTPUT_A)
motor_left    = LargeMotor(OUTPUT_D)
motor_collect = LargeMotor(OUTPUT_C)

motor_right.polarity = "inversed"
motor_left.polarity  = "inversed"

btn = Button()

stop_flag   = threading.Event()
pause_event = threading.Event()
pause_event.set()   # ikke pauset fra start

def button_watcher():
    while True:
        if btn.enter:
            if stop_flag.is_set():
                print("[ENTER] Genoptager")
                stop_flag.clear()
                pause_event.set()
            else:
                print("[ENTER] STOP")
                stop_flag.set()
                pause_event.clear()
                motor_right.off()
                motor_left.off()
            time.sleep(0.5)   # debounce

        elif btn.up:
            print("[OP] Opsamling frem")
            motor_collect.on(SpeedPercent(COLLECT_SPEED))
            while btn.up:
                time.sleep(0.05)
            motor_collect.off()

        elif btn.down:
            print("[NED] Opsamling baglens")
            motor_collect.on(SpeedPercent(-COLLECT_SPEED))
            while btn.down:
                time.sleep(0.05)
            motor_collect.off()

        time.sleep(0.05)

threading.Thread(target=button_watcher, daemon=True).start()

# ─────────────────────────────────────────────
# BEVAEGELSE
# ─────────────────────────────────────────────

def check_stop():
    """Vent hvis pauset, afbryd hvis stoppet permanent."""
    pause_event.wait()

def mm_to_rotations(mm):
    return mm / (math.pi * WHEEL_DIAM_MM)

def drive(mm):
    check_stop()
    rot = mm_to_rotations(abs(mm))
    sp  = DRIVE_SPEED if mm > 0 else -DRIVE_SPEED
    motor_right.on_for_rotations(SpeedPercent(sp), rot, block=False)
    motor_left.on_for_rotations( SpeedPercent(sp), rot, block=True)
    motor_right.off(); motor_left.off()

def turn(degrees):
    check_stop()
    if abs(degrees) < 3:
        return
    arc = (abs(degrees) / 360.0) * math.pi * WHEEL_BASE_MM
    rot = arc / (math.pi * WHEEL_DIAM_MM)
    if degrees > 0:
        motor_right.on_for_rotations(SpeedPercent( TURN_SPEED), rot, block=False)
        motor_left.on_for_rotations( SpeedPercent(-TURN_SPEED), rot, block=True)
    else:
        motor_right.on_for_rotations(SpeedPercent(-TURN_SPEED), rot, block=False)
        motor_left.on_for_rotations( SpeedPercent( TURN_SPEED), rot, block=True)
    motor_right.off(); motor_left.off()

def collect_forward():
    check_stop()
    motor_collect.on(SpeedPercent(COLLECT_SPEED))
    time.sleep(COLLECT_TIME_S)
    motor_collect.off()

def collect_reverse():
    motor_collect.on(SpeedPercent(-COLLECT_SPEED))
    time.sleep(COLLECT_TIME_S)
    motor_collect.off()

def go_to(robot_pos, heading_deg, target):
    dx   = target[0] - robot_pos[0]
    dy   = target[1] - robot_pos[1]
    dist = math.hypot(dx, dy)
    if dist < POSITION_TOL_MM:
        return target, heading_deg
    desired = math.degrees(math.atan2(dy, dx))
    diff    = (desired - heading_deg + 180) % 360 - 180
    turn(diff)
    drive(dist)
    return target, desired

# ─────────────────────────────────────────────
# KOMMANDOER
# ─────────────────────────────────────────────

def handle(cmd):
    t = cmd.get("type")

    if t == "ping":
        return {"status": "pong"}

    if t == "stop":
        stop_flag.set()
        pause_event.clear()
        motor_right.off(); motor_left.off(); motor_collect.off()
        return {"status": "stopped"}

    if t == "resume":
        stop_flag.clear()
        pause_event.set()
        return {"status": "resumed"}

    if t == "collect":
        collect_forward()
        return {"status": "collected"}

    if t == "eject":
        collect_reverse()
        return {"status": "ejected"}

    if t == "goto":
        if stop_flag.is_set():
            return {"status": "stopped"}
        robot_pos  = tuple(cmd["pos"])
        heading    = float(cmd.get("heading", 0.0))
        target     = tuple(cmd["target"])
        do_collect = cmd.get("collect", False)
        do_eject   = cmd.get("eject",   False)

        new_pos, new_heading = go_to(robot_pos, heading, target)

        if stop_flag.is_set():
            return {"status": "stopped", "pos": list(new_pos), "heading": new_heading}

        if do_collect:
            collect_forward()
        if do_eject:
            collect_reverse()

        return {
            "status":  "arrived",
            "pos":     list(new_pos),
            "heading": new_heading,
        }

    return {"status": "unknown"}

# ─────────────────────────────────────────────
# TCP SERVER
# ─────────────────────────────────────────────

def main():
    print("EV3 klar paa port {}".format(PORT))
    print("  ENTER = stop/start")
    print("  OP    = opsamling frem")
    print("  NED   = opsamling baglens")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)

    while True:
        conn, addr = srv.accept()
        print("PC forbundet: {}".format(addr))
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
                        resp = handle(cmd)
                        conn.sendall((json.dumps(resp) + "\n").encode())
                        print("  {} -> {}".format(cmd.get("type"), resp.get("status")))
                    except Exception as e:
                        err = {"status": "error", "msg": str(e)}
                        conn.sendall((json.dumps(err) + "\n").encode())
        finally:
            conn.close()

if __name__ == "__main__":
    main()