"""
robot_client.py  -  korer pa PC

Camera-guided TCP client for the EV3.
"""

import socket
import json
import math
import time

EV3_HOST = "192.168.137.3"
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

    def eject(self):
        return self._send({"type": "eject"})

    def drive_mm(self, mm, speed=None):
        cmd = {"type": "drive", "mm": float(mm)}
        if speed is not None:
            cmd["speed"] = int(speed)
        return self._send(cmd)

    def drive_to_position(self, target_mm, get_pos_fn, reverse=False,
                          tol_mm=50, speed=None, timeout=15.0, stop_fn=None):
        """
        Starts continuous drive, polls get_pos_fn() (returns (x,y) mm or None),
        sends motor_stop when within tol_mm of target_mm.
        Returns True on success, False on timeout or stop request.
        """
        if not self.connected:
            return True

        cmd = {"type": "drive_rev" if reverse else "drive_fwd"}
        if speed is not None:
            cmd["speed"] = int(speed)
        self._send(cmd)

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
                    time.sleep(0.15)
                    return True
            time.sleep(0.05)

        self._send({"type": "motor_stop"})
        print("[robot] drive_to_position timeout ({:.0f}mm from target)".format(
            math.hypot(target_mm[0] - pos[0], target_mm[1] - pos[1])
            if pos is not None else -1))
        return False

    # ── Angle helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _angle_diff(target, current):
        return (target - current + 180) % 360 - 180

    @staticmethod
    def _circular_mean(angles):
        s = sum(math.sin(math.radians(a)) for a in angles)
        c = sum(math.cos(math.radians(a)) for a in angles)
        return math.degrees(math.atan2(s, c))

    # ── Camera-guided turn ───────────────────────────────────────────────────

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

    def _turn_pass(self, target_heading, h, get_heading_fn, overshoot_comp,
                   timeout, stop_fn):
        """
        One turning pass from a known current heading `h`: pick direction,
        send the turn command, poll (3-sample circular-mean smoothing) until
        within overshoot_comp degrees of target, then stop.
        Returns True (aligned/turned), False (timeout or stop requested).
        """
        diff = self._angle_diff(target_heading, h)
        if abs(diff) < 2.0:
            return True

        sign_dir  = 1 if diff > 0 else -1
        direction = "turn_left" if diff > 0 else "turn_right"
        print("[robot] {} {:.1f}deg  ({:.1f} -> {:.1f})".format(
              direction, abs(diff), h, target_heading))
        self._send({"type": direction})

        t_start = time.time()
        history = []

        while time.time() - t_start < timeout:
            if stop_fn and stop_fn():
                self._send({"type": "motor_stop"})
                return False

            h = get_heading_fn()
            if h is not None:
                history.append(h)
                if len(history) > 3:
                    h = self._circular_mean(history[-3:])
                remaining = self._angle_diff(target_heading, h)
                if sign_dir * remaining <= overshoot_comp:
                    self._send({"type": "motor_stop"})
                    return True

            time.sleep(0.04)

        self._send({"type": "motor_stop"})
        return False

    def turn_to_heading(self, target_heading, get_heading_fn,
                        overshoot_comp=8.0, timeout=12.0, stop_fn=None,
                        tol=3.0, max_corrections=2):
        """
        Turns to target_heading with a coarse momentum-compensated pass,
        then verifies against a settled/smoothed reading and re-runs the
        same proven turning pass (same speed, same mechanism) with a small
        overshoot margin to clean up any residual error, instead of a
        separate low-speed pulse scheme (which hit motor stiction/overshoot
        at low duty cycle and made things worse).
        Returns True on success, False on timeout / no heading / stop request.
        """
        if not self.connected:
            return True

        # Wait up to 5 s for a valid heading before computing diff.
        # Never fall back to 0.0 -- that produces a completely wrong turn.
        h = None
        wait_end = time.time() + 5.0
        while time.time() < wait_end:
            h = get_heading_fn()
            if h is not None:
                break
            time.sleep(0.03)

        if h is None:
            print("[robot] No heading - skipping turn")
            return False

        if not self._turn_pass(target_heading, h, get_heading_fn,
                               overshoot_comp, timeout, stop_fn):
            if stop_fn and stop_fn():
                return False
            print("[robot] Turn timeout after {:.1f}s".format(timeout))
            return False

        for _ in range(max_corrections):
            if stop_fn and stop_fn():
                return False
            h = self._read_settled_heading(get_heading_fn)
            if h is None:
                break
            err = abs(self._angle_diff(target_heading, h))
            if err <= tol:
                return True
            print("[robot] Residual {:.1f}deg - correcting".format(err))
            if not self._turn_pass(target_heading, h, get_heading_fn,
                                   2.0, 3.0, stop_fn):
                if stop_fn and stop_fn():
                    return False
                break

        h = self._read_settled_heading(get_heading_fn)
        if h is not None:
            err = abs(self._angle_diff(target_heading, h))
            if err > tol:
                print("[robot] Turn finished, residual error {:.1f}deg".format(err))
        return True


_client = RobotClient()


def get_client():
    return _client
