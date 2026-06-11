"""
robot_client.py  -  korer paa PC

Send ET waypoint ad gangen og vent paa ACK.
Giver main.py mulighed for at genberegne ruten efter hvert stop.
"""

import socket
import json

EV3_HOST = "192.168.0.1"
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