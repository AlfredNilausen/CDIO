import cv2
import numpy as np

import white_balls as wb
import orange_balls as ob
import orange_candidates as oc


# -----------------------------
# VISUAL SETTINGS
# -----------------------------
COLOR_MAP = {
    "white": (255, 255, 255),      # BGR
    "orange": (0, 165, 255),
}

TEXT_COLOR = {
    "white": (255, 255, 255),
    "orange": (0, 165, 255),
}


def nothing(_):
    pass


def detect_all_balls(frame):
    white = wb.detect_white_balls(frame)
    orange = ob.detect_orange_balls(frame)
    balls = white + orange
    return balls, white, orange


if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("all_balls", cv2.WINDOW_NORMAL)
    cv2.namedWindow("tune_white", cv2.WINDOW_NORMAL)
    cv2.namedWindow("tune_orange", cv2.WINDOW_NORMAL)
    cv2.namedWindow("tune_candidates", cv2.WINDOW_NORMAL)

    # -----------------------------
    # WHITE SLIDERS
    # -----------------------------
    # float slider -> vi bruger *100
    cv2.createTrackbar("white_edge_thr x100", "tune_white", int(wb.EDGE_RATIO_THRESHOLD * 100), 100, nothing)
    cv2.createTrackbar("white_r_min", "tune_white", 5, 30, nothing)
    cv2.createTrackbar("white_r_max", "tune_white", 20, 40, nothing)
    cv2.createTrackbar("white_angle_step", "tune_white", wb.ANGLE_STEP, 30, nothing)

    # -----------------------------
    # ORANGE SLIDERS
    # -----------------------------
    cv2.createTrackbar("orange_area x100", "tune_orange", int(ob.ORANGE_AREA_MIN_RATIO * 100), 100, nothing)
    cv2.createTrackbar("orange_edge x100", "tune_orange", int(ob.EDGE_COVERAGE_MIN_RATIO * 100), 100, nothing)
    cv2.createTrackbar("orange_angle_step", "tune_orange", ob.ANGLE_STEP, 30, nothing)

    # neighborhood tolerance
    cv2.createTrackbar("use_neigh (0/1)", "tune_orange", 1 if ob.USE_NEIGHBORHOOD else 0, 1, nothing)
    cv2.createTrackbar("neigh_r", "tune_orange", ob.NEIGHBOR_R, 5, nothing)

    # fixed radius option
    cv2.createTrackbar("use_fixed_r (0/1)", "tune_orange", 1 if ob.USE_FIXED_R else 0, 1, nothing)
    cv2.createTrackbar("fixed_r", "tune_orange", ob.FIXED_R, 20, nothing)

    # -----------------------------
    # ORANGE CANDIDATES (Hough+edge ring) SLIDERS
    # -----------------------------
    # param2 styrer "hvor mange cirkler" (lavere = flere)
    cv2.createTrackbar("hough_acc", "tune_candidates", oc.ACC_THRESH, 40, nothing)
    cv2.createTrackbar("hough_canny", "tune_candidates", oc.CANNY_HIGH, 300, nothing)
    cv2.createTrackbar("minR", "tune_candidates", oc.MIN_R, 30, nothing)
    cv2.createTrackbar("maxR", "tune_candidates", oc.MAX_R, 40, nothing)
    cv2.createTrackbar("topN", "tune_candidates", oc.TOP_N, 60, nothing)

    # edge-ring filter
    cv2.createTrackbar("ring_in %", "tune_candidates", int(oc.RING_IN * 100), 95, nothing)
    cv2.createTrackbar("ring_out %", "tune_candidates", int(oc.RING_OUT * 100), 150, nothing)
    cv2.createTrackbar("edge_dens x1000", "tune_candidates", int(oc.EDGE_DENS_MIN * 1000), 200, nothing)
    cv2.createTrackbar("edge_pix_min", "tune_candidates", oc.EDGE_PIX_MIN, 200, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # =============================
        # READ SLIDERS -> UPDATE MODULE CONSTANTS
        # =============================

        # ----- WHITE -----
        wb.EDGE_RATIO_THRESHOLD = cv2.getTrackbarPos("white_edge_thr x100", "tune_white") / 100.0

        # radius range bruges i wb.is_white_ball; vi laver "soft override" ved at monkeypatche via nye globals
        # (så du kan bruge dem i is_white_ball hvis du ønsker at udskifte hardcoded 5..20 senere)
        white_r_min = cv2.getTrackbarPos("white_r_min", "tune_white")
        white_r_max = cv2.getTrackbarPos("white_r_max", "tune_white")
        wb.ANGLE_STEP = max(2, cv2.getTrackbarPos("white_angle_step", "tune_white"))

        # (optional) hvis du vil bruge white_r_min/max i white_balls, skal du ændre white_balls.py til at læse dem.
        # Ellers ignoreres de (se note under koden).

        # ----- ORANGE -----
        ob.ORANGE_AREA_MIN_RATIO = cv2.getTrackbarPos("orange_area x100", "tune_orange") / 100.0
        ob.EDGE_COVERAGE_MIN_RATIO = cv2.getTrackbarPos("orange_edge x100", "tune_orange") / 100.0
        ob.ANGLE_STEP = max(2, cv2.getTrackbarPos("orange_angle_step", "tune_orange"))

        ob.USE_NEIGHBORHOOD = (cv2.getTrackbarPos("use_neigh (0/1)", "tune_orange") == 1)
        ob.NEIGHBOR_R = cv2.getTrackbarPos("neigh_r", "tune_orange")

        ob.USE_FIXED_R = (cv2.getTrackbarPos("use_fixed_r (0/1)", "tune_orange") == 1)
        ob.FIXED_R = max(1, cv2.getTrackbarPos("fixed_r", "tune_orange"))

        # ----- CANDIDATES (orange_candidates.py) -----
        oc.ACC_THRESH = max(1, cv2.getTrackbarPos("hough_acc", "tune_candidates"))
        oc.CANNY_HIGH = max(1, cv2.getTrackbarPos("hough_canny", "tune_candidates"))

        oc.MIN_R = max(1, cv2.getTrackbarPos("minR", "tune_candidates"))
        oc.MAX_R = max(oc.MIN_R + 1, cv2.getTrackbarPos("maxR", "tune_candidates"))

        oc.TOP_N = max(1, cv2.getTrackbarPos("topN", "tune_candidates"))

        oc.RING_IN = cv2.getTrackbarPos("ring_in %", "tune_candidates") / 100.0
        oc.RING_OUT = cv2.getTrackbarPos("ring_out %", "tune_candidates") / 100.0
        if oc.RING_OUT <= oc.RING_IN:
            oc.RING_OUT = min(1.50, oc.RING_IN + 0.10)

        oc.EDGE_DENS_MIN = cv2.getTrackbarPos("edge_dens x1000", "tune_candidates") / 1000.0
        oc.EDGE_PIX_MIN = cv2.getTrackbarPos("edge_pix_min", "tune_candidates")

        # =============================
        # RUN DETECTORS
        # =============================
        balls, white_balls, orange_balls = detect_all_balls(frame)
        vis = frame.copy()

        for b in balls:
            cx, cy, r = b["x"], b["y"], b["r"]
            col = b.get("color", "white")
            circle_color = COLOR_MAP.get(col, (0, 255, 0))
            text_color = TEXT_COLOR.get(col, (0, 255, 0))

            cv2.circle(vis, (cx, cy), r, circle_color, 2)

            label = col
            if col == "orange":
                o = b.get("orange_ratio", None)
                e = b.get("edge_ratio", None)
                if o is not None and e is not None:
                    label = f"orange O{o:.2f} E{e:.2f}"
            elif col == "white":
                label = "white"

            cv2.putText(
                vis,
                label,
                (cx - 35, cy - r - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                1
            )

        cv2.putText(
            vis,
            f"White: {len(white_balls)} | Orange: {len(orange_balls)} | Total: {len(balls)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("all_balls", vis)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
