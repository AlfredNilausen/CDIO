import numpy as np
import cv2

def collect_zone_points(mask, zone):
    """
    zone: function (x, y, w, h) -> bool
    """
    h, w = mask.shape
    points = []

    ys, xs = np.where(mask > 0)
    for x, y in zip(xs, ys):
        if zone(x, y, w, h):
            points.append([x, y])

    return np.array(points, dtype=np.float32)

def fit_line(points):
    if len(points) < 50:
        return None

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    )
    return (vx, vy, x0, y0)

def find_fitted_lines(mask):
    h, w = mask.shape

    zones = {
        "top":    lambda x,y,w,h: y < 100,
        "bottom": lambda x,y,w,h: y > h-100,
        "left":   lambda x,y,w,h: x < 0.4*w,
        "right":  lambda x,y,w,h: x > 0.6*w,
    }

    fitted = {}

    for name, zone in zones.items():
        pts = collect_zone_points(mask, zone)
        fitted[name] = fit_line(pts)

    return fitted