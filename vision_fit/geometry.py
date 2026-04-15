import numpy as np

def intersect(line1, line2):
    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2

    A = np.array([[vx1, -vx2],
                  [vy1, -vy2]])
    B = np.array([x2 - x1,
                  y2 - y1])

    t, _ = np.linalg.lstsq(A, B, rcond=None)[0]

    x = x1 + vx1 * t
    y = y1 + vy1 * t

    return int(x), int(y)