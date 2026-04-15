import numpy as np


def _fit_y_from_x(points):
    """
    Fit model: y = m*x + b
    """
    x = points[:, 0]
    y = points[:, 1]
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def _fit_x_from_y(points):
    """
    Fit model: x = m*y + b
    """
    y = points[:, 1]
    x = points[:, 0]
    m, b = np.polyfit(y, x, 1)
    return float(m), float(b)


def _ransac_fit(points, mode="y_from_x", iterations=300, threshold=4.0, min_inliers=30):
    """
    mode = "y_from_x"  => y = m*x + b
    mode = "x_from_y"  => x = m*y + b
    """
    if points is None or len(points) < min_inliers:
        return None

    rng = np.random.default_rng()
    best_inliers = None
    best_count = 0
    best_error = 1e12

    for _ in range(iterations):
        idx = rng.choice(len(points), size=2, replace=False)
        p1 = points[idx[0]]
        p2 = points[idx[1]]

        if mode == "y_from_x":
            x1, y1 = p1
            x2, y2 = p2
            if abs(x2 - x1) < 1e-6:
                continue

            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            residuals = np.abs(points[:, 1] - (m * points[:, 0] + b))

        else:  # x_from_y
            x1, y1 = p1
            x2, y2 = p2
            if abs(y2 - y1) < 1e-6:
                continue

            m = (x2 - x1) / (y2 - y1)
            b = x1 - m * y1
            residuals = np.abs(points[:, 0] - (m * points[:, 1] + b))

        inlier_mask = residuals < threshold
        count = int(np.sum(inlier_mask))

        if count < min_inliers:
            continue

        mean_error = float(np.mean(residuals[inlier_mask]))

        if count > best_count or (count == best_count and mean_error < best_error):
            best_count = count
            best_error = mean_error
            best_inliers = points[inlier_mask]

    if best_inliers is None:
        return None

    if mode == "y_from_x":
        m, b = _fit_y_from_x(best_inliers)
    else:
        m, b = _fit_x_from_y(best_inliers)

    return {
        "mode": mode,
        "m": m,
        "b": b,
        "inliers": best_inliers
    }


def fit_frame_lines(boundary_points):
    top_model = _ransac_fit(boundary_points["top"], mode="y_from_x")
    bottom_model = _ransac_fit(boundary_points["bottom"], mode="y_from_x")
    left_model = _ransac_fit(boundary_points["left"], mode="x_from_y")
    right_model = _ransac_fit(boundary_points["right"], mode="x_from_y")

    return {
        "top": top_model,
        "bottom": bottom_model,
        "left": left_model,
        "right": right_model
    }