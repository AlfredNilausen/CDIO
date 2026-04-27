#!/usr/bin/env python3
"""
EV3 controller over Bluetooth.
Requires ev3_server.py running on the EV3 first.

Current input: keyboard
Future input:  replace keyboard_input() with camera_input()

Controls (keyboard):
    UP    arrow -> forward
    DOWN  arrow -> backward
    LEFT  arrow -> turn left  (pivot)
    RIGHT arrow -> turn right (pivot)
    DELETE      -> collector reverse (hold)
    Release key -> stop / collector normal
    ESC / q     -> quit

Requirements (PC only):
    pip install pynput
"""
import sys
import socket
import threading
from pynput import keyboard

EV3_IP   = "192.168.137.3"
EV3_PORT = 5555

# ── Connection ────────────────────────────────────────────────────────────────

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

print("Connected!\n")


def _send(cmd: bytes):
    try:
        sock.sendall(cmd)
    except OSError as e:
        print(f"Send error: {e}")


def listen_from_ev3():
    while True:
        try:
            data = sock.recv(1)
            if not data:
                break
            if data == b'X':
                print("  WARNING: Collector jammed - auto-unjamming")
        except OSError:
            break

threading.Thread(target=listen_from_ev3, daemon=True).start()


# ── Robot command API (call these from keyboard OR camera) ────────────────────

def forward():
    print("  forward")
    _send(b'F')

def backward():
    print("  backward")
    _send(b'B')

def turn_left():
    print("  left")
    _send(b'L')

def turn_right():
    print("  right")
    _send(b'R')

def stop():
    print("  stop")
    _send(b'S')

def collect_reverse():
    print("  collector: reverse")
    _send(b'M')

def collect_normal():
    print("  collector: normal")
    _send(b'N')


# ── Keyboard input (camera_input() later) ─────────────────

KEY_CMD = {
    keyboard.Key.up:     forward,
    keyboard.Key.down:   backward,
    keyboard.Key.left:   turn_left,
    keyboard.Key.right:  turn_right,
    keyboard.Key.delete: collect_reverse,
}

_active = None

def on_press(key):
    global _active
    if key == keyboard.Key.esc:
        stop()
        return False
    try:
        if key.char == 'q':
            stop()
            return False
    except AttributeError:
        pass
    fn = KEY_CMD.get(key)
    if fn and fn != _active:
        _active = fn
        fn()

def on_release(key):
    global _active
    if key == keyboard.Key.delete:
        if _active == collect_reverse:
            _active = None
            collect_normal()
        return
    if KEY_CMD.get(key) == _active:
        _active = None
        stop()

def keyboard_input():
    print("Arrow keys to drive. DELETE to reverse collector. ESC/q to quit.")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()




# TODO: replace keyboard_input() with camera_input() when ready
try:
    keyboard_input()
finally:
    stop()
    sock.close()
    print("Disconnected.")
