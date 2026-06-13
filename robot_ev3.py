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

DRIVE_SPEED        = 20
TURN_SPEED         = 20
COLLECT_SPEED      = 50
COLLECT_TIME_S     = 1.5
POSITION_TOL_MM    = 40
APPROACH_OFFSET_MM = 150   # stop this far short of a ball when collecting
TURN_STEP_DEG      = 5     # max degrees per turn increment

BOARD_WIDTH_MM     = 1680  # must match homography.py on PC
BOARD_HEIGHT_MM    = 1235
EDGE_MARGIN_MM     = 200   # reverse if estimated position is within this of any wall
EDGE_REVERSE_MM    = 200   # how far to back up when edge is detected

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
    # With both motors polarity=inversed: SpeedPercent(+) = forward, SpeedPercent(-) = backward
    sp  = DRIVE_SPEED if mm > 0 else -DRIVE_SPEED
    motor_collect.on(SpeedPercent(COLLECT_SPEED))
    motor_right.on_for_rotations(SpeedPercent( sp), rot, block=False)
    motor_left.on_for_rotations( SpeedPercent( sp), rot, block=True)
    motor_right.off(); motor_left.off()
    motor_collect.off()

def turn(degrees):
    check_stop()
    if abs(degrees) < 2:
        return
    sign       = 1 if degrees > 0 else -1
    target_rot = (abs(degrees) / 360.0) * math.pi * WHEEL_BASE_MM / (math.pi * WHEEL_DIAM_MM)
    step_rot   = (TURN_STEP_DEG / 360.0) * math.pi * WHEEL_BASE_MM / (math.pi * WHEEL_DIAM_MM)
    # Track commanded rotations instead of encoder to avoid overshoot false-exit
    done_rot   = 0.0
    while done_rot < target_rot - 0.001:
        this_rot = min(step_rot, target_rot - done_rot)
        if sign > 0:
            motor_right.on_for_rotations(SpeedPercent( TURN_SPEED), this_rot, block=False)
            motor_left.on_for_rotations( SpeedPercent(-TURN_SPEED), this_rot, block=True)
        else:
            motor_right.on_for_rotations(SpeedPercent(-TURN_SPEED), this_rot, block=False)
            motor_left.on_for_rotations( SpeedPercent( TURN_SPEED), this_rot, block=True)
        motor_right.off(); motor_left.off()
        done_rot += this_rot
        time.sleep(0.05)

def collect_forward():
    check_stop()
    motor_collect.on(SpeedPercent(COLLECT_SPEED))
    time.sleep(COLLECT_TIME_S)
    motor_collect.off()

def collect_reverse():
    motor_collect.on(SpeedPercent(-COLLECT_SPEED))
    time.sleep(COLLECT_TIME_S)
    motor_collect.off()

def _clamp_to_board(pos):
    m = EDGE_MARGIN_MM
    return (max(m, min(BOARD_WIDTH_MM - m, pos[0])),
            max(m, min(BOARD_HEIGHT_MM - m, pos[1])))

def _near_edge(pos):
    m = EDGE_MARGIN_MM
    x, y = pos
    return (x < m or x > BOARD_WIDTH_MM - m or
            y < m or y > BOARD_HEIGHT_MM - m)

def go_to(robot_pos, heading_deg, target, collecting=False):
    # Clamp target to safe zone so robot never aims at a wall
    target = _clamp_to_board(target)

    dx   = target[0] - robot_pos[0]
    dy   = target[1] - robot_pos[1]
    dist = math.hypot(dx, dy)
    drive_dist = max(0.0, dist - APPROACH_OFFSET_MM) if collecting else dist
    if drive_dist < POSITION_TOL_MM:
        return target, heading_deg
    desired = math.degrees(math.atan2(dy, dx))
    diff    = (desired - heading_deg + 180) % 360 - 180
    turn(diff)
    drive(drive_dist)

    # Estimate where the robot actually ended up
    ratio = (drive_dist / dist) if dist > 0 else 0.0
    est   = (robot_pos[0] + dx * ratio, robot_pos[1] + dy * ratio)

    # If too close to a wall, reverse along current heading to clear it
    if _near_edge(est):
        print("[edge] Naer kant - bakker {}mm".format(EDGE_REVERSE_MM))
        drive(-EDGE_REVERSE_MM)
        est = (est[0] - EDGE_REVERSE_MM * math.cos(math.radians(desired)),
               est[1] - EDGE_REVERSE_MM * math.sin(math.radians(desired)))

    return est, desired

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

        new_pos, new_heading = go_to(robot_pos, heading, target, collecting=do_collect)

        if stop_flag.is_set():
            return {"status": "stopped", "pos": list(new_pos), "heading": new_heading}

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