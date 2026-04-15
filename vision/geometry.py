import numpy as np

def intersect(l1, l2):
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2

    A = np.array([
        [x2-x1, x3-x4],
        [y2-y1, y3-y4]
    ])
    B = np.array([
        x3-x1,
        y3-y1
    ])

    t, _ = np.linalg.lstsq(A, B, rcond=None)[0]
    x = x1 + t*(x2-x1)
    y = y1 + t*(y2-y1)
    return int(x), int(y)