#!/usr/bin/env python3
"""
EV3 keyboard controller over WiFi.
Requires ev3_server.py running on the EV3 first.

Controls:
    UP    arrow -> forward
    DOWN  arrow -> backward
    LEFT  arrow -> turn left  (pivot)
    RIGHT arrow -> turn right (pivot)
    DELETE      -> motor C reverse (hold)
    Release key -> stop / motor C normal
    ESC         -> quit

Requirements (PC only):
    pip install pynput
"""
import sys
import socket
import threading
from pynput import keyboard

EV3_IP   = "192.168.137.3"
EV3_PORT = 5555

KEY_CMD = {
    keyboard.Key.up:     b'F',
    keyboard.Key.down:   b'B',
    keyboard.Key.left:   b'L',
    keyboard.Key.right:  b'R',
    keyboard.Key.space:  b'N',
    keyboard.Key.delete: b'M',
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

print("Connected! Arrow keys to drive. DELETE to reverse motor C. ESC to quit.\n")


def listen_from_ev3():
    while True:
        try:
            data = sock.recv(1)
            if not data:
                break
            if data == b'X':
                print("  WARNING: Motor C stalled!")
        except OSError:
            break

threading.Thread(target=listen_from_ev3, daemon=True).start()


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
 #   if key == keyboard.Key.delete:
 #       if _active == b'M':
 #           _active = None
  #          send(b'N')
   #         print("  motor C: normal")
    #    return
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
