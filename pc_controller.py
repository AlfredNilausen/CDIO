#!/usr/bin/env python3
"""
EV3 keyboard controller over WiFi.
Requires ev3_server.py running on the EV3 first.

Controls:
    UP    arrow -> forward
    DOWN  arrow -> backward
    LEFT  arrow -> turn left  (pivot)
    RIGHT arrow -> turn right (pivot)
    Release key -> stop
    ESC         -> quit

Requirements (PC only):
    pip install pynput
"""
import sys
import socket
from pynput import keyboard

EV3_IP   = "169.254.250.81"
EV3_PORT = 5555

KEY_CMD = {
    keyboard.Key.up:    b'F',
    keyboard.Key.down:  b'B',
    keyboard.Key.left:  b'L',
    keyboard.Key.right: b'R',
}

print(f"Connecting to EV3 at {EV3_IP}:{EV3_PORT} ...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(5)
try:
    sock.connect((EV3_IP, EV3_PORT))
    sock.settimeout(None)
except OSError as e:
    print(f"Connection failed: {e}")
    print("Is ev3_server.py running on the EV3?")
    sys.exit(1)

print("Connected! Arrow keys to drive. ESC to quit.\n")


def send(cmd: bytes):
    try:
        sock.sendall(cmd)
    except OSError as e:
        print(f"Send error: {e}")


_active = None


def on_press(key):
    global _active
    if key == keyboard.Key.esc:
        send(b'S')
        return False
    try:
        if key.char == 'q':
            send(b'S')
            print("  stop")
            return
    except AttributeError:
        pass
    cmd = KEY_CMD.get(key)
    if cmd and cmd != _active:
        _active = cmd
        send(cmd)
        print(f"  {cmd}")


def on_release(key):
    global _active
    if KEY_CMD.get(key) == _active:
        _active = None
        send(b'S')
        print("  stop")


try:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
finally:
    send(b'S')
    sock.close()
    print("Disconnected.")
