import cv2, json, os

WINDOW="preset_manager"
FILE="presets.json"

def setup_preset_manager():
    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("preset_id",WINDOW,0,100,lambda x:None)

def _load():
    if not os.path.exists(FILE): return {}
    try:
        with open(FILE) as f:
            txt=f.read().strip()
            if not txt: return {}
            return json.loads(txt)
    except: return {}

def _save(p): json.dump(p,open(FILE,"w"),indent=2)

def handle_preset_keys(k,TRACKBARS):
    presets=_load()
    pid=str(cv2.getTrackbarPos("preset_id",WINDOW))

    if k==ord("s"):
        desc=input("Beskrivelse: ")
        values={}
        for w,t in TRACKBARS:
            try: values[f"{w}::{t}"]=cv2.getTrackbarPos(t,w)
            except: pass
        presets[pid]={"name":desc,"values":values}
        _save(presets)

    if k==ord("l") and pid in presets:
        for w,t in TRACKBARS:
            key=f"{w}::{t}"
            if key in presets[pid]["values"]:
                cv2.setTrackbarPos(t,w,presets[pid]["values"][key])
