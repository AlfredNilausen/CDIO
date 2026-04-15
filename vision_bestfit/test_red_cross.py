import cv2
import numpy as np

from red_mask import red_mask
from boundary_points import extract_boundary_points
from best_fit_lines import fit_frame_lines
from geometry import draw_model_line, intersect_horizontal_vertical
from cross_detection import get_red_cross_mask, find_cross_candidate

CAMERA_INDEX = 0
SMOOTH_ALPHA = 0.80

prev_corners = None


def smooth_point(prev_pt, new_pt, alpha=0.8):
    if prev_pt is None:
        return new_pt

    x = int(round(alpha * prev_pt[0] + (1 - alpha) * new_pt[0]))
    y = int(round(alpha * prev_pt[1] + (1 - alpha) * new_pt[1]))
    return (x, y)


def draw_corner_labels(frame, corners):
    for name, (x, y) in corners.items():
        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            frame,
            name,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("red mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("cross search mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("cross mask", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # --------------------------------------------------
    # 1) Find banen
    # --------------------------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_r = red_mask(hsv)

    boundary_points = extract_boundary_points(mask_r)
    models = fit_frame_lines(boundary_points)

    draw_model_line(display, models["top"], (255, 0, 0), 2)
    draw_model_line(display, models["bottom"], (0, 0, 255), 2)
    draw_model_line(display, models["left"], (0, 255, 0), 2)
    draw_model_line(display, models["right"], (0, 255, 255), 2)

    corners = None
    cross_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    search_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if all(models[k] is not None for k in ["top", "bottom", "left", "right"]):
        new_corners = {
            "TL": intersect_horizontal_vertical(models["top"], models["left"]),
            "TR": intersect_horizontal_vertical(models["top"], models["right"]),
            "BL": intersect_horizontal_vertical(models["bottom"], models["left"]),
            "BR": intersect_horizontal_vertical(models["bottom"], models["right"]),
        }

        if all(v is not None for v in new_corners.values()):
            if prev_corners is None:
                corners = new_corners
            else:
                corners = {
                    name: smooth_point(prev_corners[name], new_corners[name], SMOOTH_ALPHA)
                    for name in new_corners
                }

            prev_corners = corners
            draw_corner_labels(display, corners)

            # --------------------------------------------------
            # 2) Find rød maske for korset i midten
            # --------------------------------------------------
            cross_mask, search_mask, _ = get_red_cross_mask(
                frame,
                corners,
                margin_px=35
            )

            # --------------------------------------------------
            # 3) Find bedste kryds-kandidat
            # --------------------------------------------------
            cross = find_cross_candidate(cross_mask, corners)

            if cross is not None:
                cnt = cross["contour"]
                cx, cy = cross["center"]
                x, y, w, h = cross["bbox"]

                # tegn kontur
                cv2.drawContours(display, [cnt], -1, (0, 255, 255), 3)

                # tegn center
                cv2.circle(display, (cx, cy), 7, (255, 255, 255), -1)

                # bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 255), 2)

                # info
                text1 = f"cross center=({cx},{cy})"
                text2 = f"area={cross['area']:.1f} extent={cross['extent']:.2f}"

                cv2.putText(
                    display,
                    text1,
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    text2,
                    (cx + 10, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    2
                )

    cv2.imshow("camera", display)
    cv2.imshow("red mask", mask_r)
    cv2.imshow("cross search mask", search_mask)
    cv2.imshow("cross mask", cross_mask)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()