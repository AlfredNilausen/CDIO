#!/usr/bin/env python3
"""
robot_ev3.py  -  korer PA EV3  (Python 3.4)

TCP server on port 9999. Receives JSON commands, drives motors.

Commands:
  ping                  -> {status: pong}
  stop                  -> emergency stop all motors + set stop flag
  motor_stop            -> stop drive motors only (collector keeps running)
  turn_left             -> start turning left  (collector on, non-blocking)
  turn_right            -> start turning right (collector on, non-blocking)
  drive   {mm: N}       -> drive N mm forward/backward (blocking, collector on)
  collect               -> run collector forward for COLLECT_TIME_S
  eject                 -> run collector backward for EJECT_TIME_S
  resume                -> clear stop flag

Buttons:
  ENTER       stop / resume toggle
  UP (hold)   collector forward
  DOWN (hold) collector reverse
"""

import math
import socket
import json
import threading
import time
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_C, OUTPUT_D, SpeedPercent
from ev3dev2.button import Button

HOST           = "0.0.0.0"
PORT           = 9999

WHEEL_BASE_MM  = 120
WHEEL_DIAM_MM  = 56

DRIVE_SPEED    = 30
BALL_SPEED     = 10   # slower when sweeping through a ball
TURN_SPEED     = 10
COLLECT_SPEED  = 30
COLLECT_TIME_S = 1.5
EJECT_TIME_S   = 20

motor_right   = LargeMotor(OUTPUT_A)
motor_left    = LargeMotor(OUTPUT_D)
motor_collect = LargeMotor(OUTPUT_C)

motor_right.polarity = "inversed"
motor_left.polarity  = "inversed"

btn = Button()

stop_flag   = threading.Event()
pause_event = threading.Event()
pause_event.set()


def _check_stop():
    pause_event.wait()


def _mm_to_rot(mm):
    return mm / (math.pi * WHEEL_DIAM_MM)


def button_watcher():
    while True:
        if btn.enter:
            if stop_flag.is_set():
                stop_flag.clear()
                pause_event.set()
                print("[ENTER] Resume")
            else:
                stop_flag.set()
                pause_event.clear()
                motor_right.off()
                motor_left.off()
                motor_collect.off()
                print("[ENTER] STOP")
            time.sleep(0.5)
        elif btn.up:
            motor_collect.on(SpeedPercent(COLLECT_SPEED))
            while btn.up:
                time.sleep(0.05)
            motor_collect.off()
        elif btn.down:
            motor_collect.on(SpeedPercent(-COLLECT_SPEED))
            while btn.down:
                time.sleep(0.05)
            motor_collect.off()
        time.sleep(0.05)


threading.Thread(target=button_watcher, daemon=True).start()


def turn_left_continuous(speed=None):
    if speed is None:
        speed = TURN_SPEED
    motor_collect.on(SpeedPercent(COLLECT_SPEED))
    motor_right.on(SpeedPercent( speed))
    motor_left.on( SpeedPercent(-speed))


def turn_right_continuous(speed=None):
    if speed is None:
        speed = TURN_SPEED
    motor_collect.on(SpeedPercent(COLLECT_SPEED))
    motor_right.on(SpeedPercent(-speed))
    motor_left.on( SpeedPercent( speed))


def drive_fwd_continuous(speed=None):
    if speed is None:
        speed = DRIVE_SPEED
    motor_collect.on(SpeedPercent(COLLECT_SPEED))
    motor_right.on(SpeedPercent( speed))
    motor_left.on( SpeedPercent( speed))


def drive_rev_continuous(speed=None):
    if speed is None:
        speed = DRIVE_SPEED
    motor_collect.on(SpeedPercent(COLLECT_SPEED))
    motor_right.on(SpeedPercent(-speed))
    motor_left.on( SpeedPercent(-speed))


def drive(mm, speed=None):
    _check_stop()
    if speed is None:
        speed = DRIVE_SPEED
    rot = _mm_to_rot(abs(mm))
    sp  = speed if mm > 0 else -speed
    motor_collect.on(SpeedPercent(COLLECT_SPEED))
    motor_right.on_for_rotations(SpeedPercent( sp), rot, block=False)
    motor_left.on_for_rotations( SpeedPercent( sp), rot, block=True)
    motor_right.off()
    motor_left.off()
    # collector stays on after drive


def handle(cmd):
    t = cmd.get("type")

    if t == "ping":
        return {"status": "pong"}

    if t == "stop":
        stop_flag.set()
        pause_event.clear()
        motor_right.off()
        motor_left.off()
        motor_collect.off()
        return {"status": "stopped"}

    if t == "motor_stop":
        motor_right.off()
        motor_left.off()
        return {"status": "stopped"}

    if t == "turn_left":
        speed = cmd.get("speed", None)
        turn_left_continuous(int(speed) if speed is not None else None)
        return {"status": "turning_left"}

    if t == "turn_right":
        speed = cmd.get("speed", None)
        turn_right_continuous(int(speed) if speed is not None else None)
        return {"status": "turning_right"}

    if t == "drive_fwd":
        speed = cmd.get("speed", None)
        drive_fwd_continuous(int(speed) if speed is not None else None)
        return {"status": "driving_fwd"}

    if t == "drive_rev":
        speed = cmd.get("speed", None)
        drive_rev_continuous(int(speed) if speed is not None else None)
        return {"status": "driving_rev"}

    if t == "drive":
        mm    = float(cmd.get("mm", 0))
        speed = cmd.get("speed", None)
        drive(mm, int(speed) if speed is not None else None)
        return {"status": "done", "mm": mm}

    if t == "collect":
        motor_collect.on(SpeedPercent(COLLECT_SPEED))
        time.sleep(COLLECT_TIME_S)
        motor_collect.off()
        return {"status": "collected"}

    if t == "eject":
        motor_collect.on(SpeedPercent(-COLLECT_SPEED))
        time.sleep(EJECT_TIME_S)
        motor_collect.off()
        return {"status": "ejected"}

    if t == "resume":
        stop_flag.clear()
        pause_event.set()
        return {"status": "resumed"}

    return {"status": "unknown"}


def main():
    print("EV3 klar pa port {}".format(PORT))
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
                try:
                    chunk = conn.recv(4096)
                except ConnectionResetError:
                    print("Client reset connection:", addr)
                    break

                if not chunk:
                    print("Client disconnected:", addr)
                    break

                buf += chunk

                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)

                    if not line.strip():
                      continue

                    try:
                       cmd = json.loads(line.decode())
                       resp = handle(cmd)
                       conn.sendall((json.dumps(resp) + "\n").encode())

                    except Exception as e:
                        print("Command error:", e)

        finally:
           conn.close()


if __name__ == "__main__":
    main()
