import cv2
import numpy as np

def create_homography(corners_px):
    """
    corners_px = [bottomLeft, topLeft, topRight, bottomRight]
    """

    pts_src = np.array(corners_px, dtype="float32")

    pts_dst = np.array([
        [0, 0],             # bottomLeft
        [0, 1235],          # topLeft
        [1680, 1235],       # topRight
        [1680, 0]           # bottomRight
    ], dtype="float32")

    H, _ = cv2.findHomography(pts_src, pts_dst)
    return H

def pixel_to_world(point_px, H):
    p = np.array([[point_px]], dtype="float32")
    world = cv2.perspectiveTransform(p, H)
    return world[0][0]