import cv2
from orange_mask import setup_orange_mask_tuner, compute_orange_mask
from ball_tuner import setup_ball_tuner, get_ball_edges_config, get_ball_closed_config
from orange_balls import setup_orange_balls_tuner, detect_orange_balls
from orange_intersection import compute_orange_intersection
from ball_closed import compute_ball_closed
from preset_manager import setup_preset_manager, handle_preset_keys

TRACKBARS=[...]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
setup_orange_mask_tuner()
setup_ball_tuner()
setup_orange_balls_tuner()
setup_preset_manager()

while True:
    ok,frame=cap.read()
    if not ok:break

    mask=compute_orange_mask(frame)
    inter=compute_orange_intersection(frame)
    closed,edges,edges_closed=compute_ball_closed(frame,get_ball_edges_config(),get_ball_closed_config(),True)

    vis=frame.copy()
    for b in detect_orange_balls(frame):
        cv2.circle(vis,(b["x"],b["y"]),b["r"],(0,165,255),2)
        cv2.putText(vis,str(b["score"]),(b["x"],b["y"]),0,0.4,(0,255,0),1)

    cv2.imshow("mask",mask)
    cv2.imshow("edges",edges)
    cv2.imshow("edges_closed",edges_closed)
    cv2.imshow("intersection",inter)
    cv2.imshow("balls",vis)

    k=cv2.waitKey(1)&0xFF
    handle_preset_keys(k,TRACKBARS)
    if k==27:break