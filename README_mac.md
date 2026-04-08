# EV3 Connection Guide — macOS

## Prerequisites

- EV3 brick running **ev3dev**
- Bluetooth paired between Mac and EV3
- Python installed on Mac with `pynput`: `pip3 install pynput`

---

## Step 1 — Pair EV3 via Bluetooth

1. On the EV3 brick: **Wireless and Networks → Bluetooth → Visible**
2. On Mac: **System Settings → Bluetooth**
3. Select your EV3 from the list and click **Connect**
4. After pairing, on the EV3: **Wireless and Networks → Bluetooth → [your Mac] → Connect**
5. OBS! **"Bluetooth on EV3 must be marked as visible"**

---

## Step 2 — Find the EV3's IP Address

In Terminal:
```
arp -a
```
Look for an IP in the `169.254.x.x` range that is not your own Mac's IP.

Your Mac's Bluetooth IP can be found with:
```
ifconfig | grep "169.254"
```
Look for the entry under a `en` or `utun` Bluetooth interface.

OR on the EV3:
**Wireless and Networks → Bluetooth → [your Mac] → Network Connection → IPv4**
Looks like `169.254.x.x`

The EV3's IP is currently set to `169.254.250.81` in `pc_controller.py`.

---

## Step 3 — Copy the Server Script to EV3 if it was changed

In Terminal from the project directory:
```
scp ev3_server.py robot@169.254.x.x:/home/robot/
```
Default credentials: **user:** `robot` **password:** `maker`

If you get a host key warning, run:
```
ssh-keygen -R 169.254.250.81
```
Then retry the `scp` command.

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

## Step 6 — Run the Controller on Mac

Be sure to change and save the EV3 IP address in `pc_controller.py`.

Open a new terminal in the project directory:
```
python3 pc_controller.py
```

> **Note:** On macOS, `pynput` may require **Accessibility permissions**. If keyboard input is not detected:
> Go to **System Settings → Privacy & Security → Accessibility** and add your Terminal app (e.g. Terminal, iTerm2, VS Code).

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
| Keyboard input not detected | Grant Accessibility permission to Terminal in System Settings |
| `scp` host key error | Run `ssh-keygen -R 169.254.250.81` then retry |
| `pynput` not found | Run `pip3 install pynput` (use `pip3`, not `pip`) |
