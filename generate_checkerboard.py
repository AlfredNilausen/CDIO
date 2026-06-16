"""
generate_checkerboard.py  -  korer pa PC

Generates a printable checkerboard image for calibrate_camera.py.
Matches calibrate_camera.py's default CHECKERBOARD = (9, 6) inner corners,
i.e. 10x7 squares.

After printing: measure one square's side with a ruler (printers rarely
scale exactly 1:1) and set SQUARE_SIZE_MM in calibrate_camera.py to the
measured value, not the nominal one below.

Run:
  python generate_checkerboard.py
Then print checkerboard.png at "actual size" / 100% scale (not "fit to
page") and tape it to something flat and rigid.
"""

import cv2
import numpy as np

SQUARES_X   = 10     # one more than calibrate_camera.py's CHECKERBOARD[0]
SQUARES_Y   = 7       # one more than calibrate_camera.py's CHECKERBOARD[1]
SQUARE_MM   = 20.0    # nominal square size - re-measure after printing
MARGIN_MM   = 15.0    # white border so the board isn't flush with the page edge
DPI         = 300
OUTPUT_FILE = "checkerboard.png"

px_per_mm = DPI / 25.4
sq_px     = int(round(SQUARE_MM * px_per_mm))
margin_px = int(round(MARGIN_MM * px_per_mm))

board_w = SQUARES_X * sq_px
board_h = SQUARES_Y * sq_px
img_w   = board_w + 2 * margin_px
img_h   = board_h + 2 * margin_px

img = np.full((img_h, img_w), 255, dtype=np.uint8)

for row in range(SQUARES_Y):
    for col in range(SQUARES_X):
        if (row + col) % 2 == 0:
            y0 = margin_px + row * sq_px
            x0 = margin_px + col * sq_px
            img[y0:y0 + sq_px, x0:x0 + sq_px] = 0

cv2.imwrite(OUTPUT_FILE, img)
print("Saved {} ({}x{} px, {:.0f} DPI, nominal {:.0f}mm squares)".format(
    OUTPUT_FILE, img_w, img_h, DPI, SQUARE_MM))
print("Print at 100% / actual size, then re-measure a square with a ruler.")
