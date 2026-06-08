import cv2
import numpy as np
from orange_intersection import compute_orange_intersection
from orange_mask import compute_orange_mask
from ball_edges import compute_ball_edges
from ball_tuner import get_ball_edges_config

WINDOW="tuner_orange_balls"

DEFAULT={
    "min_score":60,
    "edge_hit":30,
    "center_r":5,
}

def _tb(n):
    try:return cv2.getTrackbarPos(n,WINDOW)
    except cv2.error:return DEFAULT.get(n,0)

def setup_orange_balls_tuner():
    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("min_score",WINDOW,60,100,lambda x:None)
    cv2.createTrackbar("edge_hit",WINDOW,30,100,lambda x:None)
    cv2.createTrackbar("center_r",WINDOW,5,20,lambda x:None)

def detect_orange_balls(frame):
    inter = compute_orange_intersection(frame)
    orange_mask = compute_orange_mask(frame)
    edges = compute_ball_edges(frame, get_ball_edges_config())

    balls=[]
    for c in cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        (x,y),r=cv2.minEnclosingCircle(c)
        x,y,r=int(x),int(y),int(r)

        # edge hit
        circ_pts=[(int(x+r*np.cos(a)),int(y+r*np.sin(a))) for a in np.linspace(0,2*np.pi,36)]
        hits=sum(1 for px,py in circ_pts if 0<=px<edges.shape[1] and 0<=py<edges.shape[0] and edges[py,px]>0)
        edge_ratio=hits/len(circ_pts)

        # center sample
        center_mask=np.zeros_like(orange_mask)
        cv2.circle(center_mask,(x,y),_tb("center_r"),255,-1)
        center_ratio=cv2.countNonZero(center_mask & orange_mask)/max(1,cv2.countNonZero(center_mask))

        score = 50*edge_ratio + 50*center_ratio
        if score*1.0 < _tb("min_score"):
            continue

        balls.append({"x":x,"y":y,"r":r,"score":int(score)})

    return balls