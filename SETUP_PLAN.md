# GolfBot — Full Setup Plan (camera, calibration, server, movement)

Rebuild order matters: each stage assumes the previous one is solid. Don't
move to the next stage until the current one checks out — most of the
position/heading bugs so far have come from skipping straight to "drive the
robot" before the camera geometry was trustworthy.

```
1. Physical camera mount
2. Board markings (red border + cross + goals)
3. Lens calibration (checkerboard)
4. Board-corner detection tuning
5. Ball (picture) detection tuning
6. ArUco marker + heading
7. 3D pose correction (board + marker -> ground position)
8. EV3 server + TCP link
9. Movement / turning tuning
10. Route planning + full integration
11. End-to-end test checklist
```

---

## 1. Physical camera mount

This is the stage everything else depends on — fix problems here in
hardware, not in code.

- Mount the camera directly above the board, looking straight down if at
  all possible. Some tilt is fine (the math handles it), but more tilt =
  more sensitivity to every calibration error downstream. If you can get
  it within ~10-15° of straight-down, do that.
- **Bolt it down rigidly.** Once calibrated (stage 3), the camera must not
  move, rotate, refocus, or zoom — any of those invalidate the lens
  calibration and the homography.
- Frame the *entire* board with a small margin, so all 4 corners and the
  full red border are always in view, even at the corners of the camera's
  field of view (lens distortion is worst at the edges — calibration in
  stage 3 corrects for this, but only if it's not extreme).
- Use a fixed focus if your camera/lens allows it. Autofocus hunting
  between frames changes the effective focal length, which silently
  breaks the calibration from stage 3.
- Lighting: even, diffuse light across the whole board. Avoid a single
  bright lamp causing glare on one half — uneven lighting is the most
  common reason the red-border detection (stage 4) and ball/ArUco
  detection (stages 5-6) work in the middle of the frame but fail near
  the edges.
- Once mounted, measure and write down (for your report, not for the
  code): camera height above the board, and the approximate tilt angle.
  Not used directly anymore (see stage 7), but useful for sanity-checking
  later.

**Check:** open a frame in any viewer and confirm all 4 corners + full
border are inside the image with margin, in focus, evenly lit.

---

## 2. Board markings

- Red border around the playing field, continuous and high-contrast
  against the floor/background (`vision_bestfit/red_mask.py` thresholds
  on this).
- Cross marking (used by `detect_cross`) and goal openings (used by
  `detect_goals`) — both detected from the rectified `world_img`, so their
  accuracy depends entirely on stages 3+4 being correct first. Don't tune
  cross/goal detection before the corners are solid, or you'll be chasing
  a moving target.
- `BOARD_WIDTH_MM = 1680`, `BOARD_HEIGHT_MM = 1235` in
  [homography.py](vision_bestfit/homography.py#L4-L5) must match your
  actual physical board. Measure the real board and update these if they
  don't match.

**Check:** run main.py, confirm the red border is detected as 4 clean
lines (`draw_model_line` colors in the `camera` window) with no jitter
when nothing is moving.

---

## 3. Lens calibration (checkerboard)

Replaces the guessed/self-calibrated focal length with the camera's real
intrinsics (focal length + lens distortion). This is the standard,
correct way to do it — don't try to back this out from manual tape
measurements, that approach is mathematically underdetermined (confirmed
this earlier: a single corner's distance + pixel position can't pin down
focal length, there are 2 unknowns and only 1 equation).

1. Print [checkerboard.png](checkerboard.png) at **100% / actual size**
   (not "fit to page").
2. Measure one square with a ruler. Update `SQUARE_SIZE_MM` in
   [calibrate_camera.py](calibrate_camera.py#L34) to the measured value
   (printers rarely scale exactly 1:1).
3. Tape the checkerboard to something flat and rigid.
4. Run `python calibrate_camera.py`. Hold the board in front of the
   *mounted, already-positioned* camera (don't recalibrate from a
   handheld webcam and then move it to the rig).
5. Capture 15-20 views with SPACE, varying position/angle/distance/tilt
   each time, covering the whole frame including corners and edges —
   not just the center.
6. Press `c` to calibrate. Check the printed RMS reprojection error:
   **< 0.5px good, < 1.0px acceptable, anything worse → redo with more/
   better-spread captures.**
7. This produces `camera_calibration.json` (camera matrix + distortion
   coefficients).

**Not yet wired into main.py** — that's a follow-up step once this file
exists: replace the guessed/self-calibrated focal length in
`_camera_matrix()` ([main.py:305](main.py#L305-L314)) with these real
values, and apply real distortion correction (currently assumed zero).

**Check:** RMS error reported by the script is under ~1.0px.

---

## 4. Board-corner detection tuning

This is the precision bottleneck for *everything* downstream — homography,
3D board pose, marker ground projection all multiply whatever error is in
the 4 corner pixel positions. Worth getting genuinely solid before
touching anything else.

- Pipeline: `red_mask` → `extract_boundary_points` → `fit_frame_lines` →
  `intersect_horizontal_vertical` (in
  [main.py:774-795](main.py#L774-L795)).
- Watch the `edges` window and the colored line overlay in `camera`
  window. Lines should sit exactly on the red border with no wobble
  frame-to-frame.
- If corners jitter: tighten the red HSV thresholds in
  [red_mask.py](vision_bestfit/red_mask.py) first; only then consider
  raising `SMOOTH_ALPHA` (currently `0.80`,
  [main.py:32](main.py#L32)) — smoothing hides jitter but also adds lag,
  so fix detection quality before leaning on smoothing.
- Confirm the border is fully inside the frame at all 4 corners with
  margin (revisit stage 1 if not).

**Check:** with nothing moving, print the 4 corner pixel coordinates for
~100 consecutive frames and confirm they vary by at most 1-2px.

---

## 5. Ball (picture) detection tuning

`detect_balls()` ([main.py:87-127](main.py#L87-L127)) runs on the **raw**
camera frame (not the rectified `world_img`), finds circles, then
classifies each as white or orange by sampling an HSV mask under it.
Balls sit flat on the floor (no height), so unlike the ArUco marker they
don't need the 3D parallax correction from stage 7 — `pixel_to_world()`
with the plain homography is already exact for them. That means ball
position accuracy depends only on getting the circle detection itself
right, plus stage 4 (corners) for the pixel→mm conversion.

- Detection is two independent passes combined:
  - **Shape**: grayscale → blur → Canny → optional dilate → `HoughCircles`
    finds circular blobs (`canny_t1/t2`, `blur_k/sigma`, `dilate_k/it`,
    `dp`, `min_dist`, `param2`, `min_radius`/`max_radius` in
    `DetectorConfig`, [main.py:62-79](main.py#L62-L79)).
  - **Color**: HSV thresholds split each detected circle into white
    (`white_s_max` + a fixed brightness floor) vs. orange
    (`orange_h_min/max`, `orange_s_min`, `orange_v_min`) by majority pixel
    vote inside the circle.
- Tune in this order, each against real balls on the real board surface
  under your stage-1 lighting (a webcam preview on a different surface
  won't transfer):
  1. `min_radius`/`max_radius` to match the balls' actual pixel size at
     your camera height — measure a ball's pixel diameter in a captured
     frame and set the range around it.
  2. `canny_t1`/`canny_t2` and `param2` until `HoughCircles` finds real
     balls reliably without false positives on background clutter/floor
     texture (watch the `edges` window).
  3. `white_s_max`/`orange_h_min..v_min` against your actual ball colors
     under your actual lighting — color thresholds tuned at a desk under
     room light commonly fail under the rig's lighting.
- Floor/background should be a clearly different color from both ball
  types — busy or reflective floors increase both false positives (Hough)
  and color misclassification.

**Check:** place several white and orange balls around the board
(including near walls/corners) and confirm `len(whites_px)`/
`len(oranges_px)` (shown in the on-screen status, [main.py:870](main.py#L870))
match the true counts with no false detections, then confirm their drawn
positions in the `world` window land on the balls' real locations.

---

## 6. ArUco marker + heading

- `ROBOT_MARKER_ID = 0`, `MARKER_SIZE_MM = 80`
  ([main.py:35-37](main.py#L35-L37)) — confirm the printed marker is
  exactly this physical size (measure it) and is dictionary
  `DICT_4X4_50`.
- Mount the marker flat on top of the robot, parallel to the ground,
  centered as precisely as possible over the collector (stage 7 assumes
  it's directly above the point that needs to reach the ball).
- `HEADING_OFFSET_DEG = -90` calibrates "0° = East" — if your robot's
  forward direction relative to the marker's printed orientation differs,
  this is the constant to adjust. Re-derive it by placing the robot
  facing a known direction and reading the raw heading before the offset.
- Heading comes from `detect_heading()` ([main.py:351-377](main.py#L351-L377))
  via `solvePnP` on the marker's 4 corners — same camera intrinsics as
  everything else, so this also benefits directly from stage 3's real
  calibration.

**Check:** rotate the robot by hand to 0°/90°/180°/270° (using a square/
protractor against the board edges) and confirm the blue heading arrow in
the `world` window matches each time, not just one orientation.

---

## 7. 3D pose correction (board + marker → ground position)

Already implemented — this stage is "verify it," not "build it":

- `board_pose()` ([main.py:317-335](main.py#L317-L335)) solves the
  camera's pose relative to the board's 4 corners.
- `marker_ground_position()` ([main.py:338-348](main.py#L338-L348))
  takes the marker's 3D camera-frame position and projects it onto the
  board's Z=0 ground plane — this is what corrects for the marker sitting
  above the floor (parallax), and it's exact for any camera tilt as long
  as the camera intrinsics (stage 3) and corners (stage 4) are accurate.

**Check:** place the robot at a few known marked positions on the board
(e.g. taped X's with measured mm coordinates) and confirm the reported
`pos_mm` matches within a small tolerance, for robot orientations all
the way around — not just one.

---

## 8. EV3 server + TCP link

- [robot_ev3.py](robot_ev3.py) runs on the EV3 brick (`ev3dev2`, Python
  3.4), listens on `0.0.0.0:9999`, accepts newline-delimited JSON
  commands (`ping`, `drive`, `turn_left`/`turn_right`,
  `drive_fwd`/`drive_rev`, `motor_stop`, `stop`/`resume`, `collect`,
  `eject`).
- Start it on the brick: `python3 robot_ev3.py` (or via Brickman/autostart
  if you've set that up). It prints `EV3 klar pa port 9999`.
- [robot_client.py](robot_client.py) runs on the PC, connects to
  `EV3_HOST = "192.168.137.3"` ([robot_client.py:12](robot_client.py#L12))
  — update this to your EV3's actual IP (check with `ifconfig`/Brickman
  network info on the brick) before anything else will connect.
- `main.py` calls `robot.connect()` at import time
  ([main.py:23-24](main.py#L23-L24)) — `[robot] Connection failed:
  WinError 10061` just means the EV3 isn't reachable yet (wrong IP, EV3
  not running the server, or not on the same network/USB tether). The
  vision/UI side of main.py still runs fine without the robot connected,
  which is useful for testing stages 1-7 without the hardware.

**Check:** with `robot_ev3.py` running on the brick, run main.py on the
PC and confirm the console prints `[robot] Connected to <ip>:9999`
instead of the connection-failed message.

---

## 9. Movement / turning tuning

- `WHEEL_BASE_MM = 120`, `WHEEL_DIAM_MM = 56`
  ([robot_ev3.py:35-36](robot_ev3.py#L35-L36)) — measure your actual
  robot and correct these; `drive()`'s mm-to-rotation conversion depends
  directly on `WHEEL_DIAM_MM` being right.
- `turn_to_heading()` ([robot_client.py:170-212](robot_client.py#L170-L212))
  uses short pulses + re-measured settled heading rather than
  continuous-turn-until-threshold, specifically because that was found to
  overshoot inconsistently (momentum + camera/TCP latency). Tune
  `pulse_ms`/`tol` here only after stage 6/7 heading is confirmed
  accurate — if heading itself is wrong, no amount of turn-tuning will
  converge cleanly.
- `drive_to_position()` ([robot_client.py:85-140](robot_client.py#L85-L140))
  does a heading re-check/correction at the halfway point of a long
  drive, to catch wheel-slip drift. `tol_mm` (default 50) and `speed` are
  the main knobs.
- Speeds: `DRIVE_SPEED`, `TURN_SPEED`, `BALL_SPEED`, `COLLECT_SPEED`
  ([robot_ev3.py:38-41](robot_ev3.py#L38-L41)) — start low, increase once
  position tracking is confirmed accurate at low speed (errors compound
  faster at high speed and are harder to diagnose).

**Check:** issue a `drive_mm(500)` and a `turn_to_heading(90)` from a
short test script (see [tests/test_turn.py](tests/test_turn.py)) and
measure actual distance/angle moved against what was requested.

---

## 10. Route planning + full integration

- `plan_route()` ([main.py:464-503](main.py#L464-L503)) builds the
  waypoint sequence (wall-ball approach offsets, goal approach points,
  cross obstacle avoidance via `_detour`).
- `robot_executor()` ([main.py:627-757](main.py#L627-L757)) is the thread
  that walks the route, calling `robot.drive_to_position`/
  `turn_to_heading`/`collect`/`eject` in sequence, re-querying the camera
  for position/heading as it goes.
- Keys: `r` = compute+draw route, `g` = go (start executor), `s` = stop,
  `c`/`e` = manual collect/eject, `p` = status, `ESC` = quit
  ([main.py:896-944](main.py#L896-L944)).

**Check:** with one ball placed away from walls/cross, press `r` then
`g` and confirm the robot drives to it, collects, then drives to a goal
and ejects — before adding more balls/complexity.

---

## 11. End-to-end test checklist

Run through in order; if one fails, the fix is almost always in an
earlier stage, not the one that's visibly failing:

- [ ] Camera mounted rigidly, full board + margin in frame, even lighting
- [ ] `camera_calibration.json` produced with RMS < 1.0px, wired into
      `_camera_matrix()`
- [ ] Board corners stable to ~1-2px with nothing moving
- [ ] Ball detection: correct white/orange counts, no false positives,
      positions land on the real balls in the `world` window
- [ ] Heading arrow matches robot's real-world facing at 0/90/180/270°
- [ ] Marker ground position matches taped reference points around the
      whole board, all orientations
- [ ] EV3 server reachable, `[robot] Connected to ...` on startup
- [ ] `drive_mm` / `turn_to_heading` measured-accurate at low speed
- [ ] Single ball, no obstacles: full collect → goal → eject cycle works
- [ ] Multiple balls + wall balls + cross obstacle all handled
