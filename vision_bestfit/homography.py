import cv2
import numpy as np

BOARD_WIDTH_MM = 1680
BOARD_HEIGHT_MM = 1235

def compute_homographies(corners_px, display_scale=0.5):
    """
    corners_px skal være dict med:
    {
        "TL": (x, y),
        "TR": (x, y),
        "BL": (x, y),
        "BR": (x, y)
    }

    Returnerer:
    - H_px_to_world: pixel -> mm (matematisk koordinatsystem med BL=(0,0))
    - H_px_to_view:  pixel -> visningsrektangel (til warpPerspective)
    - disp_w, disp_h: størrelse på world-view vinduet
    """

    TL = corners_px["TL"]
    TR = corners_px["TR"]
    BL = corners_px["BL"]
    BR = corners_px["BR"]

    src = np.array([BL, TL, TR, BR], dtype=np.float32)

    # 1) world-koordinater i mm
    dst_world = np.array([
        [0, 0],                           # BL
        [0, BOARD_HEIGHT_MM],             # TL
        [BOARD_WIDTH_MM, BOARD_HEIGHT_MM],# TR
        [BOARD_WIDTH_MM, 0],              # BR
    ], dtype=np.float32)

    H_px_to_world = cv2.getPerspectiveTransform(src, dst_world)

    # 2) visningsvindue (OpenCV-billede: top-left origin)
    disp_w = int(round(BOARD_WIDTH_MM * display_scale))
    disp_h = int(round(BOARD_HEIGHT_MM * display_scale))

    dst_view = np.array([
        [0, disp_h - 1],          # BL nederst-venstre i view
        [0, 0],                   # TL øverst-venstre
        [disp_w - 1, 0],          # TR øverst-højre
        [disp_w - 1, disp_h - 1], # BR nederst-højre
    ], dtype=np.float32)

    H_px_to_view = cv2.getPerspectiveTransform(src, dst_view)

    return H_px_to_world, H_px_to_view, disp_w, disp_h


def pixel_to_world(point_px, H_px_to_world):
    """
    Konverterer ét pixelpunkt fra kameraet til world-koordinater i mm.
    Returnerer (x_mm, y_mm)
    """
    p = np.array([[[point_px[0], point_px[1]]]], dtype=np.float32)
    world = cv2.perspectiveTransform(p, H_px_to_world)
    x_mm = float(world[0, 0, 0])
    y_mm = float(world[0, 0, 1])
    return x_mm, y_mm


def draw_world_grid(world_img, display_scale=0.5, step_mm=200):
    """
    Tegner grid på world-view.
    """
    h, w = world_img.shape[:2]

    # lodrette grid-linjer
    for x_mm in range(0, BOARD_WIDTH_MM + 1, step_mm):
        x = int(round(x_mm * display_scale))
        cv2.line(world_img, (x, 0), (x, h - 1), (60, 60, 60), 1)
        cv2.putText(
            world_img,
            f"{x_mm}",
            (x + 4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1
        )

    # vandrette grid-linjer
    for y_mm in range(0, BOARD_HEIGHT_MM + 1, step_mm):
        y = int(round(h - 1 - y_mm * display_scale))  # world-y går opad
        cv2.line(world_img, (0, y), (w - 1, y), (60, 60, 60), 1)
        cv2.putText(
            world_img,
            f"{y_mm}",
            (5, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1
        )


def world_to_view(point_mm, display_scale=0.5):
    """
    Konverterer world-koordinater i mm til pixelposition i world-view.
    Bruges til at tegne punkter i rektificeret view.
    """
    x_mm, y_mm = point_mm
    x = int(round(x_mm * display_scale))
    y = int(round(BOARD_HEIGHT_MM * display_scale - y_mm * display_scale))
    return (x, y)