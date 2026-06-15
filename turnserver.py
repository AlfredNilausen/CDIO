#!/usr/bin/env python3
"""
turn_server_ev3.py  -  korer PA EV3

Simpel server der modtager kommandoer:
  {"type": "turn_left"}   start med at dreje til venstre
  {"type": "turn_right"}  start med at dreje til hojre
  {"type": "stop"}        stop al bevaegelse
  {"type": "drive", "mm": 300}  kor ligeud
  {"type": "ping"}

Kopiér til EV3:
  scp tests/robot/turn_server_ev3.py robot@192.168.0.1:/home/robot/

Start pa EV3:
  python3 /home/robot/turn_server_ev3.py
"""

import socket, json, math, threading
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_C, OUTPUT_D, SpeedPercent
from ev3dev2.button import Button
from ev3dev2.display import Display
from ev3dev2.sound import Sound

HOST = "0.0.0.0"
PORT = 9998   # Anden port end hoved-serveren

WHEEL_BASE_MM  = 120
WHEEL_DIAM_MM  = 56
DRIVE_SPEED    = 30
TURN_SPEED     = 20
COLLECT_SPEED  = 50

motor_right   = LargeMotor(OUTPUT_A)
motor_left    = LargeMotor(OUTPUT_D)
motor_collect = LargeMotor(OUTPUT_C)
motor_right.polarity = "inversed"
motor_left.polarity  = "inversed"

btn     = Button()
display = Display()
sound   = Sound()
ready   = threading.Event()   # sat = PC har sagt vi er klar til naeste test

def show(line1, line2=""):
    display.clear()
    display.draw.text((5, 30),  line1, fill="white")
    display.draw.text((5, 60),  line2, fill="white")
    display.update()

def button_watcher():
    """ENTER = bekraeft klar til naeste test."""
    while True:
        btn.wait_for_bump("enter")
        if not ready.is_set():
            ready.set()
            show("Klar!", "Venter pa PC...")
            sound.beep()
        import time; time.sleep(0.3)

threading.Thread(target=button_watcher, daemon=True).start()
show("Tryk ENTER", "nar du er klar")

def stop_motors():
    motor_right.off()
    motor_left.off()

def turn_left_continuous():
    """Drejer til venstre indtil stop() kaldes."""
    motor_right.on(SpeedPercent( TURN_SPEED))
    motor_left.on( SpeedPercent(-TURN_SPEED))

def turn_right_continuous():
    """Drejer til hojre indtil stop() kaldes."""
    motor_right.on(SpeedPercent(-TURN_SPEED))
    motor_left.on( SpeedPercent( TURN_SPEED))

def drive_mm(mm):
    rot = abs(mm) / (math.pi * WHEEL_DIAM_MM)
    sp  = DRIVE_SPEED if mm > 0 else -DRIVE_SPEED
    motor_right.on_for_rotations(SpeedPercent(sp), rot, block=False)
    motor_left.on_for_rotations( SpeedPercent(sp), rot, block=True)
    stop_motors()

def handle(cmd):
    t = cmd.get("type")
    if t == "ping":
        return {"status": "pong"}
    if t == "wait_ready":
        # PC spoerger: er ENTER trykket?
        if ready.is_set():
            ready.clear()
            show("Tester...", "")
            return {"status": "ready"}
        return {"status": "waiting"}
    if t == "show":
        line1 = cmd.get("line1", "")
        line2 = cmd.get("line2", "")
        show(line1, line2)
        return {"status": "ok"}
    if t == "stop":
        stop_motors()
        return {"status": "stopped"}
    if t == "turn_left":
        threading.Thread(target=turn_left_continuous, daemon=True).start()
        return {"status": "turning_left"}
    if t == "turn_right":
        threading.Thread(target=turn_right_continuous, daemon=True).start()
        return {"status": "turning_right"}
    if t == "drive":
        mm = float(cmd.get("mm", 300))
        drive_mm(mm)
        return {"status": "done", "mm": mm}
    if t == "collect":
        motor_collect.on_for_rotations(SpeedPercent(COLLECT_SPEED), 3, block=True)
        motor_collect.off()
        return {"status": "collected"}
    if t == "eject":
        motor_collect.on_for_rotations(SpeedPercent(-COLLECT_SPEED), 3, block=True)
        motor_collect.off()
        return {"status": "ejected"}
    return {"status": "unknown"}

def main():
    print("Turn server klar pa port {}".format(PORT))
    show("Server klar", "Port: {}".format(PORT))
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
                chunk = conn.recv(1024)
                if not chunk: break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip(): continue
                    try:
                        cmd  = json.loads(line.decode())
                        resp = handle(cmd)
                        conn.sendall((json.dumps(resp)+"\n").encode())
                        print("  {} -> {}".format(cmd.get("type"), resp.get("status")))
                    except Exception as e:
                        conn.sendall((json.dumps({"status":"error","msg":str(e)})+"\n").encode())
        finally:
            stop_motors()
            conn.close()

if __name__ == "__main__":
    main()