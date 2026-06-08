# orange_intersection.py
import cv2

from orange_mask import compute_orange_mask
from ball_closed import compute_ball_closed
from ball_tuner import (
    get_ball_edges_config,
    get_ball_closed_config,
)


def compute_orange_intersection(frame):
    """
    Intersection = pixels som er hvide i BÅDE:
      - orange_mask
      - edges_closed (ball_closed)

    Ingen ekstra logik.
    Ingen ekstra filtrering.
    """

    # 1. Orange farvemask
    orange_mask = compute_orange_mask(frame)

    # 2. Edges -> closed (som du allerede tuner via sliders)
    edges_closed, _, _ = compute_ball_closed(
        frame=frame,
        edges_cfg=get_ball_edges_config(),
        closed_cfg=get_ball_closed_config(),
        return_debug=False
    )

    # 3. MEGET SIMPEL intersection
    intersection = cv2.bitwise_and(orange_mask, edges_closed)

    return intersection