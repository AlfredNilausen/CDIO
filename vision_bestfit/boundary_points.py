import numpy as np


def extract_boundary_points(mask):
    """
    Returnerer randpunkter for top, bottom, left, right.
    Hver gruppe er et numpy array med shape (N, 2), dvs [x, y].
    """
    h, w = mask.shape

    # Zoner - kan justeres
    top_max_y = int(0.35 * h)
    bottom_min_y = int(0.65 * h)
    left_max_x = int(0.45 * w)
    right_min_x = int(0.55 * w)

    top_points = []
    bottom_points = []
    left_points = []
    right_points = []

    # -------- TOP / BOTTOM --------
    for x in range(w):
        # top: første røde pixel fra toppen i top-zonen
        ys_top = np.where(mask[:top_max_y, x] > 0)[0]
        if len(ys_top) > 0:
            y_top = int(ys_top.min())
            top_points.append([x, y_top])

        # bottom: sidste røde pixel fra bunden i bottom-zonen
        ys_bottom = np.where(mask[bottom_min_y:, x] > 0)[0]
        if len(ys_bottom) > 0:
            y_bottom = int(bottom_min_y + ys_bottom.max())
            bottom_points.append([x, y_bottom])

    # -------- LEFT / RIGHT --------
    for y in range(h):
        # left: første røde pixel fra venstre i left-zonen
        xs_left = np.where(mask[y, :left_max_x] > 0)[0]
        if len(xs_left) > 0:
            x_left = int(xs_left.min())
            left_points.append([x_left, y])

        # right: sidste røde pixel fra højre i right-zonen
        xs_right = np.where(mask[y, right_min_x:] > 0)[0]
        if len(xs_right) > 0:
            x_right = int(right_min_x + xs_right.max())
            right_points.append([x_right, y])

    return {
        "top": np.array(top_points, dtype=np.float32),
        "bottom": np.array(bottom_points, dtype=np.float32),
        "left": np.array(left_points, dtype=np.float32),
        "right": np.array(right_points, dtype=np.float32),
    }