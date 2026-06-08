import cv2

import white_balls as wb
import orange_ball_edge_verify as obv

# Hvis din orange pipeline bruger orange_candidates.py, tuner vi også den
try:
    import orange_candidates as oc
except Exception:
    oc = None


WIN = "all_balls"


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def nothing(_):
    pass


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_bool(x):
    return isinstance(x, bool)


# -------------------------------------------------
# Slider definition (MANUEL + AUTO)
# -------------------------------------------------
# Hver slider-def: (module, name, min, max, scale, label)
# scale: slider_value / scale -> real value
MANUAL_PARAMS = [
    # -------- WHITE --------
    (wb, "EDGE_RATIO_THRESHOLD", 0, 100, 100.0, "W EDGE%"),        # 0..1 -> %
    (wb, "ANGLE_STEP",           2, 30,   1.0,   "W ANG_STEP"),

    # Hvis du har lavet WHITE_R_MIN/MAX i white_balls.py
    (wb, "WHITE_R_MIN",          1, 30,   1.0,   "W R_MIN"),
    (wb, "WHITE_R_MAX",          1, 50,   1.0,   "W R_MAX"),

    # -------- ORANGE VERIFY --------
    (obv, "ORANGE_AREA_MIN_RATIO",     0, 100, 100.0, "O AREA%"),  # 0..1
    (obv, "EDGE_COVERAGE_MIN_RATIO",   0, 100, 100.0, "O EDGE%"),  # 0..1
    (obv, "ANGLE_STEP",                2, 30,   1.0,   "O ANG_STEP"),
    (obv, "USE_NEIGHBORHOOD",          0, 1,    1.0,   "O NEIGH_ON"),
    (obv, "NEIGHBOR_R",                0, 5,    1.0,   "O NEIGH_R"),
    (obv, "USE_FIXED_R",               0, 1,    1.0,   "O FIXR_ON"),
    (obv, "FIXED_R",                   1, 20,   1.0,   "O FIX_R"),
]

if oc is not None:
    MANUAL_PARAMS += [
        # -------- ORANGE CANDIDATES --------
        (oc, "ACC_THRESH",   1, 40,   1.0,   "C ACC"),
        (oc, "CANNY_HIGH",   1, 300,  1.0,   "C CANNY"),
        (oc, "MIN_R",        1, 25,   1.0,   "C MINR"),
        (oc, "MAX_R",        1, 40,   1.0,   "C MAXR"),
        (oc, "TOP_N",        1, 60,   1.0,   "C TOPN"),
        (oc, "RING_IN",      0, 150,  100.0, "C RIN%"),     # 0..1.5
        (oc, "RING_OUT",     0, 150,  100.0, "C ROUT%"),    # 0..1.5
        (oc, "EDGE_DENS_MIN",0, 200,  1000.0,"C DENSx1000"),
        (oc, "EDGE_PIX_MIN", 0, 200,  1.0,   "C EPIX"),
    ]


# AUTO: find andre UPPERCASE numeriske/bool parametre og giv sliders
# (for at sikre “alle variabler” uden at du skal skrive dem)
AUTO_ENABLE = True

# Navne vi ikke autogenererer (fordi de er store objekter eller ikke giver mening)
AUTO_BLOCKLIST = {"np", "cv2", "WIN", "COLOR_MAP", "TEXT_COLOR"}


def auto_params_for_module(mod, prefix):
    """
    Finder UPPERCASE (eller camel-ish) parametre i moduler og laver sliders.
    Vi prøver at gætte fornuftige ranges ud fra startværdi.
    """
    out = []
    for name, val in vars(mod).items():
        if name in AUTO_BLOCKLIST:
            continue

        # vi vil typisk kun tage konstanter/knobs
        if not (name.isupper() or name.endswith("_THRESH") or name.endswith("_RATIO") or name.endswith("_MIN") or name.endswith("_MAX")):
            continue

        if is_bool(val):
            out.append((mod, name, 0, 1, 1.0, f"{prefix} {name[:10]}"))
            continue

        if not is_number(val):
            continue

        # gæt range
        if isinstance(val, float):
            # mange ratios ligger 0..1
            if 0.0 <= val <= 1.0:
                out.append((mod, name, 0, 100, 100.0, f"{prefix} {name[:10]}%"))
            # små floats (densities)
            elif 0.0 <= val <= 0.2:
                out.append((mod, name, 0, 200, 1000.0, f"{prefix} {name[:10]}"))
            else:
                # generel float
                vmax = int(max(1.0, min(500.0, val * 3)))
                out.append((mod, name, 0, vmax, 1.0, f"{prefix} {name[:10]}"))
        else:
            # int: giv et spænd omkring værdien
            vmax = int(max(5, min(500, val * 4 if val > 0 else 50)))
            out.append((mod, name, 0, vmax, 1.0, f"{prefix} {name[:10]}"))

    return out


def build_param_list():
    params = []

    # Manuel liste først (har pæne labels)
    for p in MANUAL_PARAMS:
        mod, name, lo, hi, scale, label = p
        if hasattr(mod, name):
            params.append(p)

    if AUTO_ENABLE:
        # Tilføj auto for hver modul
        params += auto_params_for_module(wb, "W")
        params += auto_params_for_module(obv, "O")
        if oc is not None:
            params += auto_params_for_module(oc, "C")

    # Fjern dubletter (samme modul+name)
    seen = set()
    uniq = []
    for p in params:
        key = (p[0].__name__, p[1])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    return uniq


PARAMS = build_param_list()


# -------------------------------------------------
# Trackbars
# -------------------------------------------------
def setup_trackbars():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # Opret trackbars på samme vindue
    for (mod, name, lo, hi, scale, label) in PARAMS:
        val = getattr(mod, name)
        if is_bool(val):
            init = 1 if val else 0
        else:
            init = int(val * scale)
        init = clamp(init, lo, hi)
        cv2.createTrackbar(label, WIN, init, hi, nothing)


def apply_trackbars():
    for (mod, name, lo, hi, scale, label) in PARAMS:
        raw = cv2.getTrackbarPos(label, WIN)

        current = getattr(mod, name)
        if is_bool(current):
            setattr(mod, name, raw == 1)
        else:
            value = raw / scale
            # hvis det oprindeligt var int, behold int
            if isinstance(current, int) and not isinstance(current, bool):
                value = int(round(value))
            setattr(mod, name, value)

    # lille sanity: hvis vi har min/max par
    if hasattr(wb, "WHITE_R_MIN") and hasattr(wb, "WHITE_R_MAX"):
        if wb.WHITE_R_MAX <= wb.WHITE_R_MIN:
            wb.WHITE_R_MAX = wb.WHITE_R_MIN + 1


# -------------------------------------------------
# Detection
# -------------------------------------------------
def detect_all_balls(frame):
    balls = []
    balls.extend(wb.detect_white_balls(frame))
    balls.extend(obv.detect_orange_balls(frame))
    return balls


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    setup_trackbars()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        apply_trackbars()

        balls = detect_all_balls(frame)
        vis = frame.copy()

        for b in balls:
            x, y, r = b["x"], b["y"], b["r"]
            color = b.get("color", "unknown")

            if color == "white":
                col = (255, 255, 255)
            elif color == "orange":
                col = (0, 140, 255)
            else:
                col = (0, 255, 0)

            cv2.circle(vis, (x, y), r, col, 2)
            cv2.putText(
                vis,
                color,
                (x - 15, y - r - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                col,
                1
            )

        cv2.putText(
            vis,
            f"balls: {len(balls)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(WIN, vis)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
