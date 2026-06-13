# CDIO Robot System — Handoff Document

## Overview

The system is split into two parts: a **PC** running computer vision and route planning,
and an **EV3 robot** running a motor controller. They communicate over a TCP socket
via a Bluetooth network connection.

```
PC                                      EV3
──────────────────────────────          ──────────────────────────
main.py          (vision + routing)     robot_ev3.py  (motor control)
robot_client.py  (sends commands)  <──> TCP socket port 9999
vision_bestfit/  (vision modules)       calibrate.py  (calibration tool)
```

---

## Files

### `main.py` (PC)

**What it does:**
- Opens the overhead camera and detects the red border of the track using
  line fitting and homography to produce a corrected top-down world view in mm.
- Detects the red cross in the centre of the track (defines the quadrant divider).
- Detects white and orange balls using Hough circle detection.
- Detects the robot position and heading using an ArUco marker (ID 0) on the robot.
- Plans a collection route: quadrant by quadrant (Q1->Q2->Q3->Q4), white balls
  first within each quadrant, with detour logic around the cross and
  perpendicular wall-approach for edge balls.
- Sends waypoints one at a time to the robot via robot_client.py in a
  background thread. After each waypoint is reached, the route is
  recalculated from the new robot position and current ball detections.

**What works:**
- Track border detection and world-view homography
- Cross detection and quadrant splitting
- Ball detection (white and orange) in world mm coordinates
- ArUco heading detection (marker ID 0, DICT_4X4_50)
- Route planning with cross exclusion zone and wall approach logic
- Route visualisation in the world-view window
- Background robot executor thread (sends one waypoint, waits for ACK,
  recalculates, continues)
- Auto-recalculate route every 3 seconds

**What still needs work / known issues:**
- `robot` is not imported in the current repo version. Add these three lines
  after the imports at the top:
      from robot_client import get_client
      robot = get_client()
      robot.connect()
- Ball detection sometimes picks up false positives (reflections, cross arms).
  Tune DetectorConfig parameters, especially param2 (lower = more circles
  detected, higher = fewer but more reliable).
- The ArUco heading offset (+90 / -90) may need adjusting depending on how
  the marker is physically mounted. Change _rvec_to_heading in main.py.
- The route recalculation after each waypoint replaces the entire remaining
  route. A smarter version would only recalculate if a ball has actually
  disappeared from the camera (i.e. been collected).
- No mechanism yet to detect whether a ball was actually collected.
  Currently relies on the camera seeing fewer balls after each stop.
- CAMERA_INDEX = 1 -- change to 0 if the overhead camera is the default device.

**Key bindings:**
  r    Recalculate route from current detections
  g    Go -- start robot execution
  s    Stop robot immediately
  c    Run collector motor forward once
  e    Eject -- run collector motor in reverse once
  p    Print status snapshot to terminal
  ESC  Quit

---

### `robot_client.py` (PC)

**What it does:**
Thin TCP client that connects to the EV3 and sends commands as
newline-delimited JSON. Used by main.py to send one waypoint at a time
and wait for the response.

**What works:**
- connect() / disconnect()
- ping()            checks the connection is alive
- stop()            sends emergency stop
- resume()          resumes after stop
- collect()         triggers collector motor forward
- eject()           triggers collector motor in reverse
- send_waypoint()   sends a single goto command and blocks until EV3
                    replies "arrived" or "stopped"

**Configuration to change:**
  EV3_HOST = "192.168.0.1"   # EV3 IP over Bluetooth tether
  EV3_PORT = 9999
  TIMEOUT  = 30              # seconds -- increase if robot is slow

**What still needs work:**
- No reconnection logic. If the socket drops mid-run, send_waypoint returns
  None and the executor thread stops. Add auto-reconnect for robustness.
- No heartbeat / keepalive. Long pauses between waypoints can cause the
  connection to time out silently.

---

### `robot_ev3.py` (EV3)

**What it does:**
TCP server running on the EV3. Receives JSON commands, drives motors,
and reports back. Runs a background button-watcher thread so the
physical EV3 buttons work at all times.

**What works:**
- TCP server on port 9999, handles one connection at a time
- ping / stop / resume / collect / eject commands
- goto command: turns to face target heading, drives straight to target,
  replies "arrived" with updated position and heading
- Physical buttons:
    ENTER        stop / resume toggle
    UP (hold)    collector motor forward
    DOWN (hold)  collector motor reverse

**Configuration to change (top of file):**
  WHEEL_BASE_MM  = 120   # distance between wheel centres -- MEASURE ON ROBOT
  WHEEL_DIAM_MM  = 56    # wheel diameter -- MEASURE ON ROBOT
  DRIVE_SPEED    = 30    # % of max speed while driving straight
  TURN_SPEED     = 20    # % of max speed while turning in place
  COLLECT_SPEED  = 50    # % for collector motor
  COLLECT_TIME_S = 1.5   # seconds collector runs per ball

**Motor ports (already set correctly):**
  OUTPUT_A  right drive wheel     polarity: normal
  OUTPUT_D  left drive wheel      polarity: inversed (motor mounted backwards)
  OUTPUT_C  collector mechanism

**What still needs work / known issues:**
- Turning accuracy depends entirely on WHEEL_BASE_MM being correct.
  Use calibrate.py to tune before a real run.
- No encoder feedback during straight drive. If a wheel slips the position
  estimate drifts. Consider reading motor.position after each move.
- No collision avoidance. Robot will drive into the wall if given a
  waypoint outside the track bounds.
- Single connection only. If main.py crashes and reconnects, the EV3 server
  accepts the new connection but any command sent while a motor is still
  running is ignored until the motor finishes.
- EV3 runs Python 3.4 -- no f-strings, no X|Y type hints.
  Keep all string formatting as .format().

---

### `calibrate.py` (EV3)

**What it does:**
Interactive command-line calibration tool. Run directly on the EV3 via SSH
to tune WHEEL_BASE_MM and WHEEL_DIAM_MM before a real run.

**Usage:**
  ssh robot@192.168.0.1
  python3 /home/robot/calibrate.py

**Commands:**
  d 500    Drive forward 500 mm
  b 200    Reverse 200 mm
  l 90     Turn left 90 degrees
  r 90     Turn right 90 degrees
  q        Quit

**Calibration procedure:**
  1. Mark a start line on the floor.
  2. Run "d 500" and measure actual distance.
       Too short -> increase WHEEL_DIAM_MM
       Too long  -> decrease WHEEL_DIAM_MM
  3. Run "l 360" -- robot should complete exactly one full rotation.
       Under-rotates -> increase WHEEL_BASE_MM
       Over-rotates  -> decrease WHEEL_BASE_MM
  4. Copy the confirmed values to the top of robot_ev3.py.

**What works:**
- Forward / reverse driving with calibration hints
- Left / right turns with calibration hints

**What still needs work:**
- No collector motor test command (add "c" for forward, "e" for reverse)
- WHEEL_BASE_MM / WHEEL_DIAM_MM must be edited manually in the file

---

## Deployment

**One-time setup:**
  1. Pair EV3 via Bluetooth. EV3 IP is 192.168.0.1 (tether interface).
  2. Copy files to EV3:
       scp robot_ev3.py  robot@192.168.0.1:/home/robot/
       scp calibrate.py  robot@192.168.0.1:/home/robot/
  3. Set EV3_HOST = "192.168.0.1" in robot_client.py.

**Every run:**
  Terminal 1 (SSH to EV3):
    python3 /home/robot/robot_ev3.py

  Terminal 2 (PC):
    cd C:\Users\...\CDIO
    python main.py

**Startup sequence:**
  1. EV3 server starts, prints "EV3 klar paa port 9999"
  2. main.py starts, connects to EV3, opens camera windows
  3. Wait for track + cross to be detected (borders appear in camera view)
  4. Press r -- route is calculated and shown in world view
  5. Press g -- robot starts driving

---

## What Has Been Most Helpful

- **Homography / world-view**: converting everything to mm coordinates early
  on made route planning, distance calculation and ball positions much more
  reliable than working in pixels.

- **Point-cloud + minAreaRect for border detection**: more robust than trying
  to find a closed 4-sided contour when one edge has low contrast or is
  partially outside the frame.

- **Recursive _detour() around the cross**: checking sub-segments recursively
  ensures the path never clips the exclusion zone even when a waypoint is
  close to the cross. The push direction uses the midpoint-to-cross vector
  so it always pushes outward correctly.

- **One-waypoint-at-a-time protocol**: sending waypoints individually and
  recalculating after each one means the route automatically adapts if the
  camera sees fewer balls or the robot position drifts. It also makes stop/
  resume trivial since the robot just stops after the current waypoint.

- **Nearest-neighbour per quadrant**: simple and fast, good enough for the
  number of balls involved. No need for a full TSP solver.

- **ArUco markers for heading**: far more reliable than inferring heading from
  wheel encoders alone, especially given the lack of a gyro sensor. Heading
  offset (-90 degrees currently) may need adjusting based on marker orientation.

- **Wall approach logic**: perpendicular approach to edge balls prevents the
  robot from trying to drive parallel to a wall to reach a ball, which would
  be unreliable given positional drift. Corner balls get a diagonal approach.
