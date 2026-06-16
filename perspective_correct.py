"""
perspective_correct.py

Korrigerer robotposition for parallax-fejl.

ArUco-markoren sidder DIREKTE OVER opsamleren.
Gulvposition G = (Mx, My) i verdensrum -- Z ignoreres.

Kamera (C)
    |
    M  <- ArUco markoren (hoejde h over gulvet)
    |  <- lodret
    G  <- opsamler / gulvposition
-------- gulvplan Z=0

Kamerakalibrering:
  Vi kender de fysiske afstande fra kameraet til hvert hjorne (maalt med
  maalbaand). Det giver os en langt mere praecis kameramatrix end at gaette
  med f = max(W, H).

Maaling:
  Maalt fra kameralinsen (eller kamerahuset) til hvert hjorne i mm.
  Hvis kameraet haenger lodret ned: alle 4 afstande er ens (= kamerahoejde).
  Hvis kameraet er skraastillet: de 4 afstande er forskellige.

  Eksempel:
    DIST_TL_MM = 1450   # afstand fra kamera til TL-hjorne
    DIST_TR_MM = 1200   # afstand fra kamera til TR-hjorne
    DIST_BL_MM = 1300   # osv.
    DIST_BR_MM = 1100

  Disse bruges til at kalibrere den effektive broendevidde (focal length)
  for hvert hjorne, og vi finder den bedste gennemsnitlige focal length
  ved mindste kvadrater.
"""

import cv2
import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────────────
# FYSISKE AFSTANDE (mm) -- MAALT MED MAALBAAND
# Maales fra kameralinsen til hvert hjorne af banen
# ─────────────────────────────────────────────────────────────────────────────
DIST_TL_MM = 2080.0   # <- top venstre hjorne
DIST_TR_MM = 2000.0  # <- top hoejre hjorne
DIST_BL_MM = 1920.0  # <- bund venstre hjorne
DIST_BR_MM = 1880.0 # <- bund hoejre hjorne
# ─────────────────────────────────────────────────────────────────────────────


def _board_obj_points(board_w_mm, board_h_mm):
    return np.array([
        [0,          board_h_mm, 0],   # TL
        [board_w_mm, board_h_mm, 0],   # TR
        [0,          0,          0],   # BL
        [board_w_mm, 0,          0],   # BR
    ], dtype=np.float32)


def calibrate_focal_length(corners_px, board_w_mm, board_h_mm,
                            dist_tl, dist_tr, dist_bl, dist_br,
                            img_w, img_h):
    """
    Estimerer focal length fra maalte afstande til hjornerne.

    Princip: for et hjorne med kendt verden-koordinat P_world og
    maalt afstand d til kameraet ved vi at:
        d = ||tvec + R * P_world||

    Men en enklere approksimation der virker godt:
    Kameraet ser hvert hjorne i pixel (u,v). Den normaliserede
    retning fra billedcentrum er (u-cx, v-cy). Vi ved den fysiske
    afstand d. Den 3D-vektor i kamerarum har laengde d.

    Focal length f opfylder:
        f = sqrt(d^2 - (u-cx)^2 - (v-cy)^2)
    naar den enhed der bruges er pixels og mm er ens skala.

    Vi bruger i stedet den korrekte metode:
    Med kendte verdenskoordinater og maalt afstand kan vi opsaette
    solvePnP med en initial focal length og iterere.
    Returner bedste focal length estimate.
    """
    cx = img_w / 2.0
    cy = img_h / 2.0

    corners_list = ["TL", "TR", "BL", "BR"]
    dists        = [dist_tl, dist_tr, dist_bl, dist_br]
    world_pts    = _board_obj_points(board_w_mm, board_h_mm)
    img_pts      = np.array([corners_px[n] for n in corners_list], dtype=np.float32)

    focal_estimates = []

    for j, (name, d) in enumerate(zip(corners_list, dists)):
        u, v = img_pts[j]
        Xw, Yw, Zw = world_pts[j]  # Zw = 0 (gulvplan)

        # Normaliserede billedkoordinater (uden focal length endnu)
        xn = (u - cx)
        yn = (v - cy)

        # Med pinhole-model: den 3D-punkt i kamerarum har afstand d.
        # Vi har: X_cam^2 + Y_cam^2 + Z_cam^2 = d^2
        # og:     X_cam / Z_cam = xn / f,  Y_cam / Z_cam = yn / f
        # => Z_cam^2 * (1 + xn^2/f^2 + yn^2/f^2) = d^2
        #
        # Ommform: f^2 * d^2 / Z_cam^2 = f^2 + xn^2 + yn^2
        #
        # Vi kan ikke loeser dette direkte uden Z_cam, men vi kan
        # finde et godt estimat ved at antage Z_cam ~ d * cos(theta)
        # hvor theta er vinklen fra den optiske akse.
        #
        # Iterativ loesning:
        f_est = max(img_w, img_h)   # start-gaet
        for _ in range(20):
            # Z_cam fra pinhole med nuvaerende f
            denom = math.sqrt(1.0 + (xn/f_est)**2 + (yn/f_est)**2)
            Z_cam = d / denom
            # Ny focal length: f = xn * Z_cam / X_cam ... men vi kender
            # ikke X_cam direkte. Brug afstandsligningen direkte:
            # f^2 = (d^2 - Z_cam^2) * Z_cam^2 / (xn^2 + yn^2)
            # hvis xn^2 + yn^2 > 0
            r2 = xn**2 + yn**2
            if r2 < 1e-6:
                f_new = d   # hjorne er naer optisk akse
            else:
                f_new = math.sqrt(max(1.0, (d**2 - Z_cam**2)) / r2) * Z_cam
            if abs(f_new - f_est) < 0.1:
                break
            f_est = 0.5 * f_est + 0.5 * f_new   # daemp oscillation

        focal_estimates.append(f_est)
        print("  {} afstand={:.0f}mm  pixel=({:.0f},{:.0f})  f_est={:.1f}px".format(
              name, d, u, v, f_est))

    # Brug medianen -- robust over for outliers
    f_median = float(np.median(focal_estimates))
    print("  Focal length (median): {:.1f} px".format(f_median))
    return f_median


def make_camera_matrix(img_w, img_h, focal_length=None,
                       corners_px=None, board_w_mm=None, board_h_mm=None,
                       dist_tl=None, dist_tr=None, dist_bl=None, dist_br=None):
    """
    Laver kameramatrix.

    Hvis afstande er givet kalibreres focal length fra maalte afstande.
    Ellers bruges f = max(img_w, img_h) som groest estimat.

    Eksempel med afstande:
        cam = make_camera_matrix(1280, 720,
                                 corners_px=corners,
                                 board_w_mm=1500, board_h_mm=1200,
                                 dist_tl=1400, dist_tr=1200,
                                 dist_bl=1300, dist_br=1100)
    """
    cx = img_w / 2.0
    cy = img_h / 2.0

    if (focal_length is None and corners_px is not None
            and all(d is not None for d in [dist_tl, dist_tr, dist_bl, dist_br])):
        print("[kalibrering] Beregner focal length fra maalte afstande...")
        focal_length = calibrate_focal_length(
            corners_px, board_w_mm, board_h_mm,
            dist_tl, dist_tr, dist_bl, dist_br,
            img_w, img_h)
    elif focal_length is None:
        focal_length = max(img_w, img_h)
        print("[kalibrering] Bruger estimeret f={:.0f} px (ingen afstande givet)".format(focal_length))

    return np.array([
        [focal_length, 0,            cx],
        [0,            focal_length, cy],
        [0,            0,            1 ],
    ], dtype=np.float64)


def compute_camera_pose(corners_px, board_w_mm, board_h_mm,
                        camera_matrix, dist_coeffs=None):
    """
    Beregner kameraets pose fra banens 4 hjorner.
    Returnerer (R, t) eller (None, None).
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    obj_pts = _board_obj_points(board_w_mm, board_h_mm)
    img_pts = np.array([corners_px[n] for n in ["TL","TR","BL","BR"]],
                       dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, None

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec


def marker_to_ground(tvec_aruco, R_board, t_board):
    """
    Finder gulvpositionen direkte under markoren.
    = markorens (X, Y) i verdensrum, Z ignoreres.
    """
    tvec_aruco = np.array(tvec_aruco, dtype=np.float64).reshape(3, 1)
    marker_world = (R_board.T @ (tvec_aruco - t_board)).flatten()
    return float(marker_world[0]), float(marker_world[1])


def make_aruco_obj_pts(marker_size_mm):
    h = marker_size_mm / 2.0
    return np.array([
        [-h,  h, 0], [ h,  h, 0],
        [ h, -h, 0], [-h, -h, 0],
    ], dtype=np.float32)


def get_corrected_robot_pose(frame, detector, obj_pts_aruco,
                              camera_matrix, corners_px,
                              board_w_mm, board_h_mm,
                              heading_offset=0.0,
                              marker_id=0,
                              dist_coeffs=None):
    """
    Returnerer korrigeret robotpose.

    Returnerer:
      {"found": True,
       "heading": float,          -- grader (0=hoejre, 90=op)
       "pos_mm": (x, y),          -- gulvposition under markoren
       "pos_px_raw": (x, y),      -- raa pixel-center (til debug)
       "marker_height_mm": float} -- estimeret markorhoejde
    eller
      {"found": False, "reason": str}
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    R_board, t_board = compute_camera_pose(
        corners_px, board_w_mm, board_h_mm, camera_matrix, dist_coeffs)
    if R_board is None:
        return {"found": False, "reason": "solvePnP paa banehjorner fejlede"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {"found": False, "reason": "ingen ArUco markorer"}

    for i, mid in enumerate(ids.flatten()):
        if mid != marker_id:
            continue
        mc = corners[i][0]
        ok, rvec_m, tvec_m = cv2.solvePnP(
            obj_pts_aruco, mc.astype(np.float32),
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue

        R_m, _ = cv2.Rodrigues(rvec_m)
        heading = math.degrees(math.atan2(-R_m[1, 0], R_m[0, 0])) + heading_offset

        gx, gy = marker_to_ground(tvec_m, R_board, t_board)

        tvec_arr     = np.array(tvec_m, dtype=np.float64).reshape(3, 1)
        marker_world = (R_board.T @ (tvec_arr - t_board)).flatten()

        return {
            "found"            : True,
            "heading"          : heading,
            "pos_mm"           : (gx, gy),
            "pos_px_raw"       : (int(mc[:,0].mean()), int(mc[:,1].mean())),
            "marker_height_mm" : float(marker_world[2]),
        }

    return {"found": False, "reason": "ID {} ikke fundet".format(marker_id)}