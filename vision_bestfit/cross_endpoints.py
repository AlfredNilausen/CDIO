import cv2
import numpy as np


def transform_points(points_xy, H):
    """
    Transformerer punkter med en homografi.

    points_xy: numpy array shape (N, 2)
    H: 3x3 homografi
    """
    pts = np.array(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)


def world_to_pixel(point_world, H_px_to_world):
    """
    Konverterer world-koordinat -> kamera-pixel
    ved at invertere homografien.
    """
    H_world_to_px = np.linalg.inv(H_px_to_world)
    p = np.array([[point_world]], dtype=np.float32)
    px = cv2.perspectiveTransform(p, H_world_to_px)
    return float(px[0, 0, 0]), float(px[0, 0, 1])


def skeletonize(binary_img):
    """
    Morfologisk skeletonization.
    Input: binært billede (0/255)
    Output: skeleton (0/255)
    """
    img = binary_img.copy()
    skel = np.zeros(img.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel


def find_skeleton_endpoints(skel):
    """
    Finder endpoints i skeleton.
    Et endpoint har præcis 1 nabo i 8-neighborhood.

    Returnerer liste af (x, y).
    """
    endpoints = []
    h, w = skel.shape

    binary = (skel > 0).astype(np.uint8)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if binary[y, x] == 0:
                continue

            neighborhood = binary[y - 1:y + 2, x - 1:x + 2]
            count = int(np.sum(neighborhood)) - 1  # minus centerpixel

            if count == 1:
                endpoints.append((x, y))

    return endpoints


def cluster_points(points, cluster_dist=12):
    """
    Simpel clustering af nærliggende punkter.
    Returnerer cluster-centre som liste af (x, y).
    """
    if len(points) == 0:
        return []

    pts = [np.array(p, dtype=np.float32) for p in points]
    used = [False] * len(pts)
    clusters = []

    for i in range(len(pts)):
        if used[i]:
            continue

        group = [pts[i]]
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(pts)):
                if used[j]:
                    continue

                for g in group:
                    if np.linalg.norm(pts[j] - g) < cluster_dist:
                        group.append(pts[j])
                        used[j] = True
                        changed = True
                        break

        center = np.mean(group, axis=0)
        clusters.append((float(center[0]), float(center[1])))

    return clusters


def order_endpoints_by_angle(points, center):
    """
    Nummerér endpoints som endpoint1..4 efter vinkel omkring center.
    Sorterer stigende atan2.
    """
    cx, cy = center
    pts_with_angle = []

    for p in points:
        x, y = p
        angle = np.arctan2(y - cy, x - cx)
        pts_with_angle.append((angle, (float(x), float(y))))

    pts_with_angle.sort(key=lambda t: t[0])

    ordered = {}
    for idx, (_, pt) in enumerate(pts_with_angle, start=1):
        ordered[f"endpoint{idx}"] = pt

    return ordered


def contour_center_world(contour_px, H_px_to_world):
    """
    Beregn center af kontur i world-koordinater.
    """
    contour_px_2d = contour_px.reshape(-1, 2).astype(np.float32)
    contour_world = transform_points(contour_px_2d, H_px_to_world)

    cx = float(np.mean(contour_world[:, 0]))
    cy = float(np.mean(contour_world[:, 1]))
    return cx, cy


def compute_cross_endpoints_from_contour(contour_px, H_px_to_world):
    """
    Finder 4 endpoints for korset.

    Returnerer:
    {
        "center_world": (x_mm, y_mm),
        "endpoints_world": {
            "endpoint1": (...),
            "endpoint2": (...),
            "endpoint3": (...),
            "endpoint4": (...)
        },
        "endpoints_camera": {
            "endpoint1": (...),
            "endpoint2": (...),
            "endpoint3": (...),
            "endpoint4": (...)
        },
        "skeleton_world": skeleton_img,
        "local_bbox_world": (min_x, min_y, max_x, max_y)
    }
    """
    if contour_px is None or len(contour_px) < 10:
        return None

    # 1) Transformér hele konturen til world
    contour_px_2d = contour_px.reshape(-1, 2).astype(np.float32)
    contour_world = transform_points(contour_px_2d, H_px_to_world)

    # 2) Lav et lokalt world-billede omkring konturen
    min_x = int(np.floor(np.min(contour_world[:, 0]))) - 20
    max_x = int(np.ceil(np.max(contour_world[:, 0]))) + 20
    min_y = int(np.floor(np.min(contour_world[:, 1]))) - 20
    max_y = int(np.ceil(np.max(contour_world[:, 1]))) + 20

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    if width <= 0 or height <= 0:
        return None

    local_contour = contour_world.copy()
    local_contour[:, 0] -= min_x
    local_contour[:, 1] = max_y - contour_world[:, 1]  # vend y så billede går nedad

    local_contour_int = np.round(local_contour).astype(np.int32).reshape(-1, 1, 2)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [local_contour_int], -1, 255, thickness=-1)

    # 3) Skeleton
    skel = skeletonize(mask)

    # 4) Endpoint-kandidater
    raw_endpoints = find_skeleton_endpoints(skel)
    clustered = cluster_points(raw_endpoints, cluster_dist=10)

    if len(clustered) < 4:
        return None

    # Hvis der er flere end 4 clusters, vælg de 4 fjerneste fra center
    center_local_x = float(np.mean(local_contour[:, 0]))
    center_local_y = float(np.mean(local_contour[:, 1]))

    clustered_sorted = sorted(
        clustered,
        key=lambda p: np.hypot(p[0] - center_local_x, p[1] - center_local_y),
        reverse=True
    )

    selected = clustered_sorted[:4]

    # 5) Konvertér tilbage til world-koordinater
    endpoints_world_list = []
    for lx, ly in selected:
        wx = lx + min_x
        wy = max_y - ly
        endpoints_world_list.append((float(wx), float(wy)))

    # 6) Center i world
    center_world = contour_center_world(contour_px, H_px_to_world)

    # 7) Nummerér endpoint1..4 efter vinkel omkring center
    endpoints_world = order_endpoints_by_angle(endpoints_world_list, center_world)

    # 8) Map world-endpoints tilbage til kamera-pixels
    endpoints_camera = {
        name: world_to_pixel(pt, H_px_to_world)
        for name, pt in endpoints_world.items()
    }

    return {
        "center_world": center_world,
        "endpoints_world": endpoints_world,
        "endpoints_camera": endpoints_camera,
        "skeleton_world": skel,
        "local_bbox_world": (min_x, min_y, max_x, max_y)
    }