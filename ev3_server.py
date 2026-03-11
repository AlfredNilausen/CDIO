#!/usr/bin/env python3
"""
Run this on the EV3 via SSH:
    python3 ev3_server.py

Uses only pre-installed ev3dev2 libraries — nothing to install.
Motors: A = left wheel, D = right wheel
"""
import socket
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_D, SpeedPercent

SPEED = 60
PORT  = 5555

try:
    motor_left  = LargeMotor(OUTPUT_A)
    print("Motor A (left) OK")
except Exception as e:
    print("Motor A ERROR: {}".format(e))
    raise

try:
    motor_right = LargeMotor(OUTPUT_D)
    print("Motor D (right) OK")
except Exception as e:
    print("Motor D ERROR: {}".format(e))
    raise

def forward():  motor_left.on(SpeedPercent( SPEED)); motor_right.on(SpeedPercent( SPEED))
def backward(): motor_left.on(SpeedPercent(-SPEED)); motor_right.on(SpeedPercent(-SPEED))
def left():     motor_left.on(SpeedPercent(-SPEED)); motor_right.on(SpeedPercent( SPEED))
def right():    motor_left.on(SpeedPercent( SPEED)); motor_right.on(SpeedPercent(-SPEED))
def stop():     motor_left.off(); motor_right.off()

COMMANDS = {b'F': forward, b'B': backward, b'L': left, b'R': right, b'S': stop}

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
    client.close()
    server.close()
