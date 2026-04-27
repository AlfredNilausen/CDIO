#!/usr/bin/env python3
"""
Run this on the EV3 via SSH:
    python3 ev3_server.py

Uses only pre-installed ev3dev2 libraries — nothing to install.
Motors: A = left wheel, D = right wheel
"""
import socket
import time
import threading
from ev3dev2.motor import LargeMotor, MediumMotor, OUTPUT_A, OUTPUT_C, OUTPUT_D, SpeedPercent

SPEED = 60
PORT  = 5555

try:
    motor_left  = LargeMotor(OUTPUT_D)
    print("Motor D (left) OK")
except Exception as e:
    print("Motor D ERROR: {}".format(e))
    raise

try:
    motor_right = LargeMotor(OUTPUT_A)
    print("Motor A (right) OK")
except Exception as e:
    print("Motor A ERROR: {}".format(e))
    raise

try:
    motor_c = LargeMotor(OUTPUT_C)
  #  motor_c.on(SpeedPercent(SPEED))
    print("Motor C (center) OK - running")
except Exception as e:
    print("Motor C ERROR: {}".format(e))
    raise

def forward():       motor_left.on(SpeedPercent( SPEED)); motor_right.on(SpeedPercent( SPEED))
def backward():      motor_left.on(SpeedPercent(-SPEED)); motor_right.on(SpeedPercent(-SPEED))
def left():          motor_left.on(SpeedPercent(-SPEED)); motor_right.on(SpeedPercent( SPEED))
def right():         motor_left.on(SpeedPercent( SPEED)); motor_right.on(SpeedPercent(-SPEED))
def stop():          motor_left.off(); motor_right.off()
def motor_c_rev():   motor_c.on(SpeedPercent(-SPEED))
def motor_c_norm():  motor_c.on(SpeedPercent( SPEED))

_unjamming = False

def unjam_motor_c():
    global _unjamming
    if _unjamming:
        return
    _unjamming = True
    print("Motor C jammed - reversing to unjam...")
    motor_c.on(SpeedPercent(-SPEED))
    time.sleep(0.5)
    motor_c.on(SpeedPercent(SPEED))
    print("Motor C unjammed")
    _unjamming = False

def stall_monitor():
    while True:
        try:
            if motor_c.is_stalled:
                unjam_motor_c()
        except Exception:
            pass
        time.sleep(0.1)

threading.Thread(target=stall_monitor, daemon=True).start()

COMMANDS = {b'F': forward, b'B': backward, b'L': left, b'R': right, b'S': stop,
            b'M': motor_c_rev, b'N': motor_c_norm}

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('', PORT))
server.listen(1)
print("Listening on port {} ...".format(PORT))

client, addr = server.accept()
print("Connected: {}".format(addr))

try:
    while True:
        data = client.recv(1)
        if not data:
            break
        if data in COMMANDS:
            try:
                COMMANDS[data]()
            except Exception as e:
                print("Command error: {}".format(e))
finally:
    stop()
    motor_c.off()
    client.close()
    server.close()
