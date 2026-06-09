import cv2
import cv2.aruco as aruco
import numpy as np
 
# ─────────────────────────────────────────────
# CONFIG  — edit these to match your setup
# ─────────────────────────────────────────────
 
CAMERA_INDEX   = 1          # 0 = built-in webcam, 1 = USB camera
MARKER_ID      = 0          # which ArUco ID is on the robot
MARKER_SIZE_MM = 80         # physical side length of printed marker in mm
 
# ArUco dictionary — try DICT_4X4_50 first (matches the printed marker)
# If detection fails, try DICT_6X6_50 or DICT_5X5_50
ARUCO_DICT = aruco.DICT_4X4_50
 
 
# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
 
def make_camera_matrix(w, h):
    """
    Rough camera matrix when you don't have a calibration file.
    Good enough for heading angles; replace with real values for accuracy.
    """
    focal = max(w, h)
    return np.array([
        [focal,     0, w / 2],
        [    0, focal, h / 2],
        [    0,     0,     1],
    ], dtype=np.float64)
 
 
def rotation_to_heading(rvec):
    """
    Convert an ArUco rotation vector to a 2-D heading angle.
 
    ArUco marker Z-axis points out of the paper toward the camera.
    The marker's X-axis is its 'forward' direction.
    We project X-axis onto the image plane to get heading.
 
    Returns angle in degrees: 0 = right, 90 = up, 180/-180 = left, -90 = down
    """
    R, _ = cv2.Rodrigues(rvec)
 
    # X-axis of the marker in camera space
    x_axis_cam = R[:, 0]          # first column
 
    # Project onto image plane (ignore Z depth)
    dx = x_axis_cam[0]
    dy = x_axis_cam[1]            # positive Y is DOWN in image coords
 
    # atan2 gives angle from positive-X axis, going counter-clockwise
    angle_rad = np.arctan2(-dy, dx)   # negate dy so up = positive
    angle_deg = np.degrees(angle_rad)
 
    return angle_deg
 
 
def heading_to_cardinal(angle_deg):
    """Map a heading angle to a compass label."""
    # Normalize to 0-360
    a = angle_deg % 360
 
    directions = [
        (0,    "E"),
        (45,   "NE"),
        (90,   "N"),
        (135,  "NW"),
        (180,  "W"),
        (225,  "SW"),
        (270,  "S"),
        (315,  "SE"),
        (360,  "E"),
    ]
    closest = min(directions, key=lambda d: abs(d[0] - a))
    return closest[1]
 
 
def draw_heading_arrow(frame, center, angle_deg, length=60):
    """Draw a direction arrow on the frame."""
    angle_rad = np.radians(angle_deg)
    ex = int(center[0] + length * np.cos(angle_rad))
    ey = int(center[1] - length * np.sin(angle_rad))   # screen Y is flipped
 
    cv2.arrowedLine(frame, center, (ex, ey),
                    color=(0, 255, 0), thickness=3, tipLength=0.3)
 
 
def draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, size):
    """Draw X/Y/Z axes on the marker for visual debugging."""
    axis_len = size * 0.5
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                      rvec, tvec, axis_len)
 
 
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
 
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
    # ArUco setup
    aruco_dict   = aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = aruco.DetectorParameters()
    detector     = aruco.ArucoDetector(aruco_dict, aruco_params)
 
    # 3-D corner points of the marker (in marker's own coordinate system)
    half = MARKER_SIZE_MM / 2.0
    obj_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float32)
 
    last_heading  = None
    last_cardinal = "?"
    last_center   = None
 
    print("Press ESC to quit.")
    print("Detected headings will print here.\n")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — check CAMERA_INDEX")
            break
 
        H, W = frame.shape[:2]
        camera_matrix = make_camera_matrix(W, H)
        dist_coeffs   = np.zeros((5, 1), dtype=np.float64)
 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
 
        found = False
 
        if ids is not None:
            # Draw all detected markers lightly
            aruco.drawDetectedMarkers(frame, corners, ids)
 
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id != MARKER_ID:
                    continue
 
                found = True
                marker_corners = corners[i][0]   # shape (4, 2)
 
                # Pose estimation
                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    marker_corners.astype(np.float32),
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
 
                if not success:
                    continue
 
                # Heading
                heading  = rotation_to_heading(rvec)
                cardinal = heading_to_cardinal(heading)
 
                # Marker center in image
                cx = int(marker_corners[:, 0].mean())
                cy = int(marker_corners[:, 1].mean())
                center = (cx, cy)
 
                last_heading  = heading
                last_cardinal = cardinal
                last_center   = center
 
                # Draw axes and arrow
                draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE_MM)
                draw_heading_arrow(frame, center, heading)
 
                # Label on marker
                cv2.putText(frame,
                            f"ID {marker_id}  {heading:+.1f}deg  {cardinal}",
                            (cx - 60, cy - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
                # Print to terminal every frame (you can throttle this later)
                print(f"\rHeading: {heading:+7.1f} deg   Cardinal: {cardinal:2s}   "
                      f"Center: ({cx}, {cy})        ", end="", flush=True)
 
        # HUD overlay
        status_color = (0, 255, 0) if found else (0, 0, 255)
        status_text  = (f"Heading: {last_heading:+.1f} deg  ({last_cardinal})"
                        if last_heading is not None else "No marker detected")
        cv2.putText(frame, status_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
 
        if not found:
            cv2.putText(frame, "Looking for marker...", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
 
        cv2.imshow("Direction detector", frame)
 
        if cv2.waitKey(1) == 27:   # ESC
            break
 
    print()   # newline after \r prints
    cap.release()
    cv2.destroyAllWindows()
 
 
# ─────────────────────────────────────────────
# Integration helper — call this from other code
# ─────────────────────────────────────────────
 
def get_robot_direction(frame, camera_matrix=None, dist_coeffs=None):
    """
    Given a single camera frame, returns a dict:
      {
        'found':    bool,
        'heading':  float,   # degrees, 0=right 90=up 180=left -90=down
        'cardinal': str,     # N / NE / E / SE / S / SW / W / NW
        'center':   (x, y),  # pixel position of marker center
      }
 
    Use this when integrating with ball_tracker_v4.py:
 
        from direction import get_robot_direction
        result = get_robot_direction(frame)
        if result['found']:
            heading = result['heading']
    """
    H, W = frame.shape[:2]
    if camera_matrix is None:
        camera_matrix = make_camera_matrix(W, H)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
 
    half = MARKER_SIZE_MM / 2.0
    obj_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float32)
 
    aruco_dict   = aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = aruco.DetectorParameters()
    detector     = aruco.ArucoDetector(aruco_dict, aruco_params)
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
 
    if ids is None:
        return {'found': False, 'heading': None, 'cardinal': None, 'center': None}
 
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id != MARKER_ID:
            continue
 
        marker_corners = corners[i][0]
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            marker_corners.astype(np.float32),
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            continue
 
        heading  = rotation_to_heading(rvec)
        cardinal = heading_to_cardinal(heading)
        cx = int(marker_corners[:, 0].mean())
        cy = int(marker_corners[:, 1].mean())
 
        return {
            'found':    True,
            'heading':  heading,
            'cardinal': cardinal,
            'center':   (cx, cy),
        }
 
    return {'found': False, 'heading': None, 'cardinal': None, 'center': None}
 
 
if __name__ == "__main__":
    main()