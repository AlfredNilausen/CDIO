"""
robot_client.py  —  kører på PC

Sender waypoints fra main.py til EV3'en via TCP socket.
Importer denne i main.py og kald send_route().

Bluetooth-netværk:
  - Forbind EV3 til PC via Bluetooth
  - ev3dev opretter automatisk et netværksinterface (bnep0 eller lignende)
  - Standard EV3 IP over BT:  10.42.0.2  (eller se via: ssh robot@ev3dev.local)
  - Standard EV3 IP over USB: 192.168.0.1
"""

import socket
import json
import threading
import time

# ─────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────

EV3_HOST = "192.168.0.1"    # ← skift til din EV3's IP
                           #   BT:  10.42.0.2  (typisk)
                           #   USB: 192.168.0.1
EV3_PORT = 9999
TIMEOUT  = 10              # sekunder


class RobotClient:
    def __init__(self, host=EV3_HOST, port=EV3_PORT):
        self.host    = host
        self.port    = port
        self.sock    = None
        self.connected = False

    def connect(self):
        """Opret forbindelse til EV3. Returnerer True hvis OK."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(TIMEOUT)
            self.sock.connect((self.host, self.port))
            self.connected = True
            print(f"[robot] Forbundet til EV3 på {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[robot] Kunne ikke forbinde: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.sock:
            self.sock.close()
            self.sock = None
        self.connected = False

    def _send(self, cmd: dict) -> dict | None:
        """Send én kommando og vent på svar."""
        if not self.connected:
            print("[robot] Ikke forbundet")
            return None
        try:
            msg = (json.dumps(cmd) + "\n").encode()
            self.sock.sendall(msg)
            # Læs svar (newline-termineret JSON)
            buf = b""
            while b"\n" not in buf:
                chunk = self.sock.recv(1024)
                if not chunk:
                    break
                buf += chunk
            resp = json.loads(buf.split(b"\n")[0].decode())
            return resp
        except Exception as e:
            print(f"[robot] Send-fejl: {e}")
            self.connected = False
            return None

    def ping(self) -> bool:
        resp = self._send({"type": "ping"})
        return resp is not None and resp.get("status") == "pong"

    def stop(self):
        self._send({"type": "stop"})

    def send_route(self, waypoints_mm: list, robot_pos_mm: tuple,
                   heading_deg: float) -> bool:
        """
        Send en liste waypoints til EV3.

        waypoints_mm : liste af (x_mm, y_mm) tupler
        robot_pos_mm : aktuel robot-position (x_mm, y_mm)
        heading_deg  : aktuel robot-retning i grader (0=højre, 90=op)

        Returnerer True når EV3 melder "done".
        """
        if not waypoints_mm:
            return True

        cmd = {
            "type":      "waypoints",
            "waypoints": [[float(x), float(y)] for x,y in waypoints_mm],
            "robot_pos": [float(robot_pos_mm[0]), float(robot_pos_mm[1])],
            "heading":   float(heading_deg),
        }

        print(f"[robot] Sender {len(waypoints_mm)} waypoints...")
        resp = self._send(cmd)

        if resp and resp.get("status") == "done":
            print(f"[robot] Rute færdig. Slutposition: {resp.get('final_pos')}")
            return True
        else:
            print(f"[robot] Uventet svar: {resp}")
            return False


# ─────────────────────────────────────────────
# Singleton der bruges fra main.py
# ─────────────────────────────────────────────

_client = RobotClient()

def get_client() -> RobotClient:
    return _client


# ─────────────────────────────────────────────
# Hurtig test (kør denne fil direkte)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    c = RobotClient()
    if c.connect():
        print("Ping:", c.ping())
        # Test-rute: kør en lille firkant
        test_route = [(200,200),(400,200),(400,400),(200,400),(200,200)]
        c.send_route(test_route, robot_pos_mm=(200,200), heading_deg=0)
        c.disconnect()