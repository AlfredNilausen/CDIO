import socket
import json
import math
import time

EV3_HOST = "192.168.0.1"
EV3_PORT = 9999
TIMEOUT  = 30

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
            print("[robot] Connected to {}:{}".format(self.host, self.port))
            return True
        except Exception as e:
            print("[robot] Connection failed: {}".format(e))
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
            print("[robot] Send error: {}".format(e))
            self.connected = False
            return None

    def ping(self):
        r = self._send({"type": "ping"})
        return r is not None and r.get("status") == "pong"

    def stop(self):
        self._send({"type": "stop"})

    def motor_stop(self):
        self._send({"type": "motor_stop"})

    def resume(self):
        self._send({"type": "resume"})

    def collect(self):
        return self._send({"type": "collect"})

    def eject(self, speed=100, duration=1000.0):
        return self._send({"type": "eject", "speed": int(speed), "duration": float(duration)})

    def drive_mm(self, mm, speed=None):
        cmd = {"type": "drive", "mm": float(mm)}
        if speed is not None:
            cmd["speed"] = int(speed)
        return self._send(cmd)

    def drive_to_position(self, target_mm, get_pos_fn, reverse=False,
                          tol_mm=50, speed=20, timeout=5.0, stop_fn=None,
                          get_heading_fn=None):
        if not self.connected:
            return True

        cmd = {"type": "drive_rev" if reverse else "drive_fwd"}
        if speed is not None:
            cmd["speed"] = int(speed)
        self._send(cmd)

        pos0  = get_pos_fn()
        dist0 = math.hypot(target_mm[0] - pos0[0], target_mm[1] - pos0[1]) if pos0 else None
        corrected = get_heading_fn is None

        t_start = time.time()
        while time.time() - t_start < timeout:
            if stop_fn and stop_fn():
                self._send({"type": "motor_stop"})
                return False
            pos = get_pos_fn()
            if pos is not None:
                d = math.hypot(target_mm[0] - pos[0], target_mm[1] - pos[1])
                if d <= tol_mm:
                    self._send({"type": "motor_stop"})
                    time.sleep(0.05)
                    return True

                if not corrected and dist0 is not None and d <= dist0 / 2.0:
                    corrected = True
                    self._send({"type": "motor_stop"})
                    dx, dy  = target_mm[0] - pos[0], target_mm[1] - pos[1]
                    desired = math.degrees(math.atan2(dy, dx))
                    if reverse:
                        desired = (desired + 180) % 360
                    self.turn_to_heading(desired, get_heading_fn, pulse_ms=100, stop_fn=stop_fn)
                    if stop_fn and stop_fn():
                        return False
                    self._send(cmd)

            time.sleep(0.05)

        self._send({"type": "motor_stop"})
        print("[robot] drive_to_position timeout ({:.0f}mm from target)".format(
            math.hypot(target_mm[0] - pos[0], target_mm[1] - pos[1])
            if pos is not None else -1))
        return False

    @staticmethod
    def _angle_diff(target, current):
        return (target - current + 180) % 360 - 180

    @staticmethod
    def _circular_mean(angles):
        s = sum(math.sin(math.radians(a)) for a in angles)
        c = sum(math.cos(math.radians(a)) for a in angles)
        return math.degrees(math.atan2(s, c))

    def _read_settled_heading(self, get_heading_fn, samples=3, gap=0.03):
        """Smoothed heading after a brief settle, to avoid reacting to a
        single noisy frame (e.g. vibration right after a motor stop)."""
        time.sleep(0.2)
        vals = []
        for _ in range(samples):
            h = get_heading_fn()
            if h is not None:
                vals.append(h)
            time.sleep(gap)
        if not vals:
            return None
        return self._circular_mean(vals) if len(vals) > 1 else vals[0]

    def turn_to_heading(self, target_heading, get_heading_fn,
                        pulse_ms=100, tol=3.0, timeout=2.0, stop_fn=None):
        if not self.connected:
            return True

        t_start = time.time()
        while time.time() - t_start < timeout:
            if stop_fn and stop_fn():
                self._send({"type": "motor_stop"})
                return False

            h = self._read_settled_heading(get_heading_fn)
            if h is None:
                print("[robot] No heading - skipping turn")
                return False

            diff = self._angle_diff(target_heading, h)
            if abs(diff) <= tol:
                return True

            direction = "turn_left" if diff > 0 else "turn_right"
            scale   = min(1.0, abs(diff) / 20.0)
            pulse_s = max(0.03, (pulse_ms / 1000.0) * scale)
            self._send({"type": direction})
            time.sleep(pulse_s)
            self._send({"type": "motor_stop"})

        h = self._read_settled_heading(get_heading_fn)
        err = abs(self._angle_diff(target_heading, h)) if h is not None else -1
        print("[robot] Turn timeout after {:.1f}s, residual {:.1f}deg".format(timeout, err))
        return False


_client = RobotClient()


def get_client():
    return _client
