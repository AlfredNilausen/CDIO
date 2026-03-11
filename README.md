# EV3 Connection Guide

## Prerequisites

- EV3 brick running **ev3dev**
- Bluetooth paired between PC and EV3
- Python installed on PC with `pynput`: `pip install pynput`

---

## Step 1 — Pair EV3 via Bluetooth

1. On the EV3 brick: **Wireless and Networks → Bluetooth → Visible**
2. On Windows: **Settings → Bluetooth & devices → Add device**
3. Select your EV3 and pair it
4. After pairing, on the EV3: **Wireless and Networks → Bluetooth → [your PC] → Connect**
5. OBS! **"Bluetooth on EV3 marked as visible"**

---

## Step 2 — Find the EV3's IP Address

In PowerShell:
```
arp -a
```
Look for an IP in the `169.254.x.x` range that is not your own PC's IP.
Your PC's Bluetooth IP can be found with `ipconfig` under **Bluetooth Network Connection**.

OR on EV3
**Wireless and Networks → Bluetooth → [your PC] → Network Connection ->IPv4**
Looks like `169.254.x.x`



The EV3's IP is currently set to `169.254.250.81` in `pc_controller.py`.

---

## Step 3 — Copy the Server Script to EV3 if it was changed

In PowerShell from the project directory:
```
scp ev3_server.py robot@169.254.250.81:/home/robot/
```
Default credentials: **user:** `robot` **password:** `maker`

---

## Step 4 — SSH into the EV3

```
ssh robot@169.254.xx.xx
```
**password:** `maker`
---

## Step 5 — Run the Server on the EV3

In the SSH session:
```
python3 ~/ev3_server.py
```

Expected output:
```
Motor A (left) OK
Motor D (right) OK
Listening on port 5555 ...
```

Leave this running and keep the SSH session open.

---

## Step 6 — Run the Controller on PC

Be sure to change and save EV3 IP adrasse in pc_controller.py
Open a new terminal in the project directory:
```
python pc_controller.py
```

Expected output:
```
Connecting to EV3 at 169.254.250.81:5555 ...
Connected! Arrow keys to drive. ESC to quit.
```

---

## Controls

| Key | Action |
|-----|--------|
| Arrow Up | Forward |
| Arrow Down | Backward |
| Arrow Left | Turn left (pivot) |
| Arrow Right | Turn right (pivot) |
| Release key | Stop |
| ESC | Quit |

---

## Changing the EV3 IP

If the EV3's IP changes, update line 21 in `pc_controller.py`:
```python
EV3_IP = "169.254.x.x"  # set to your EV3's actual IP
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on SSH | EV3 Bluetooth network not connected — repeat Step 1 |
| `Connection failed` in controller | Check EV3 IP, ensure `ev3_server.py` is running |
| Motor error on EV3 | Check motor cables are in ports A (left) and D (right) |
| `arp -a` shows no EV3 | Re-connect Bluetooth network on EV3 brick |