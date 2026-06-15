"""
robot_client.py  -  korer paa PC

Send ET waypoint ad gangen og vent paa ACK.
Giver main.py mulighed for at genberegne ruten efter hvert stop.
"""

import socket
import json
import math
import time
EV3_HOST = "192.168.0.1"
##EV3_HOST = "192.168.0.1"
EV3_PORT = 9999
TIMEOUT  = 30   # sekunder - lang nok til at robotten kan koere

class RobotClient:
    def __init__(self, host=EV3_HOST, port=EV3_PORT):
        self.host      = host
        self.port      = port
        self.sock      = None
        self.connected = False

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(TIMEOUT)
            self.sock.connect((self.host, self.port))
            self.connected = True
            print("[robot] Forbundet til {}:{}".format(self.host, self.port))
            return True
        except Exception as e:
            print("[robot] Kunne ikke forbinde: {}".format(e))
            self.connected = False
            return False

    def disconnect(self):
        if self.sock:
            self.sock.close()
            self.sock = None
        self.connected = False

    def _send(self, cmd):
        if not self.connected:
            return None
        try:
            self.sock.sendall((json.dumps(cmd) + "\n").encode())
            buf = b""
            while b"\n" not in buf:
                chunk = self.sock.recv(1024)
                if not chunk:
                    break
                buf += chunk
            return json.loads(buf.split(b"\n")[0].decode())
        except Exception as e:
            print("[robot] Fejl: {}".format(e))
            self.connected = False
            return None

    def ping(self):
        r = self._send({"type": "ping"})
        return r is not None and r.get("status") == "pong"

    def stop(self):
        self._send({"type": "stop"})

    def resume(self):
        self._send({"type": "resume"})

    def collect(self):
        return self._send({"type": "collect"})

    def eject(self):
        return self._send({"type": "eject"})

    # ── Camera-guided turn helpers ──────────────────────────────────────────

    @staticmethod
    def _angle_diff(target, current):
        return (target - current + 180) % 360 - 180

    @staticmethod
    def _circular_mean(angles):
        sin_sum = sum(math.sin(math.radians(a)) for a in angles)
        cos_sum = sum(math.cos(math.radians(a)) for a in angles)
        return math.degrees(math.atan2(sin_sum, cos_sum))

    def turn_left(self):
        return self._send({"type": "turn_left"})

    def turn_right(self):
        return self._send({"type": "turn_right"})

    def drive_mm(self, mm, collecting=False):
        return self._send({"type": "drive", "mm": float(mm), "collect": collecting})

    def turn_to_heading(self, target_heading, get_heading_fn,
                        overshoot_comp=5.0, timeout=10.0, stop_fn=None):
        """
        Camera-guided turn.
        Sends turn_left/turn_right, polls get_heading_fn() until heading
        is within overshoot_comp degrees of target, then sends stop.
        Returns True on success, False on timeout or external stop.
        """
        if not self.connected:
            return True

        h = get_heading_fn()
        if h is None:
            h = 0.0
        diff = self._angle_diff(target_heading, h)
        if abs(diff) < 1.0:
            return True

        sign_dir  = 1 if diff > 0 else -1
        direction = "turn_left" if diff > 0 else "turn_right"
        self._send({"type": direction})

        start   = time.time()
        history = []

        while time.time() - start < timeout:
            if stop_fn and stop_fn():
                self._send({"type": "motor_stop"})
                return False

            h = get_heading_fn()
            if h is not None:
                history.append(h)
                if len(history) > 4:
                    h = self._circular_mean(history[-4:])
                remaining = self._angle_diff(target_heading, h)
                if sign_dir * remaining <= overshoot_comp:
                    self._send({"type": "motor_stop"})
                    time.sleep(0.3)
                    return True

            time.sleep(0.12)

        self._send({"type": "motor_stop"})
        print("[robot] Turn timeout na {:.1f}s".format(timeout))
        return False

    def send_waypoint(self, target_mm, robot_pos_mm, heading_deg, do_collect=False):
        """
        Send ET waypoint. Blokerer til robotten melder 'arrived' eller 'stopped'.
        Returnerer dict: {"status": ..., "pos": [...], "heading": ...}
        """
        cmd = {
            "type":    "goto",
            "target":  [float(target_mm[0]),    float(target_mm[1])],
            "pos":     [float(robot_pos_mm[0]),  float(robot_pos_mm[1])],
            "heading": float(heading_deg),
            "collect": do_collect,
        }
        print("[robot] Sender waypoint {}".format(target_mm))
        return self._send(cmd)


_client = RobotClient()

def get_client():
    return _client