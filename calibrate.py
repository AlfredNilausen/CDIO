#!/usr/bin/env python3
"""
calibrate.py  -  Koer paa EV3 via SSH

Tester korslen og drejning saa du kan kalibrere
WHEEL_BASE_MM og WHEEL_DIAM_MM i robot_ev3.py

Start:  python3 /home/robot/calibrate.py
"""

import math
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_D, SpeedPercent

# ── Disse skal matche robot_ev3.py ──────────────
WHEEL_BASE_MM  = 750
WHEEL_DIAM_MM  = 65
DRIVE_SPEED    = 30
TURN_SPEED     = 20
# ────────────────────────────────────────────────

motor_right = LargeMotor(OUTPUT_A)
motor_left  = LargeMotor(OUTPUT_D)
motor_right.polarity = "inversed"
motor_left.polarity  = "inversed"

def stop():
    motor_right.off()
    motor_left.off()

def mm_to_rotations(mm):
    return mm / (math.pi * WHEEL_DIAM_MM)

def drive(mm):
    rot = mm_to_rotations(abs(mm))
    sp  = DRIVE_SPEED if mm > 0 else -DRIVE_SPEED
    motor_right.on_for_rotations(SpeedPercent(sp), rot, block=False)
    motor_left.on_for_rotations( SpeedPercent(sp), rot, block=True)
    stop()

def turn(degrees):
    arc = (abs(degrees) / 360.0) * math.pi * WHEEL_BASE_MM
    rot = arc / (math.pi * WHEEL_DIAM_MM)
    sp  = TURN_SPEED
    if degrees > 0:   # venstre
        motor_right.on_for_rotations(SpeedPercent( sp), rot, block=False)
        motor_left.on_for_rotations( SpeedPercent(-sp), rot, block=True)
    else:             # hojre
        motor_right.on_for_rotations(SpeedPercent(-sp), rot, block=False)
        motor_left.on_for_rotations( SpeedPercent( sp), rot, block=True)
    stop()

def menu():
    print("\n============================")
    print(" KALIBRERING")
    print(" WHEEL_BASE_MM = {}".format(WHEEL_BASE_MM))
    print(" WHEEL_DIAM_MM = {}".format(WHEEL_DIAM_MM))
    print("============================")
    print(" d <mm>    - koer ligeud  (ex: d 500)")
    print(" b <mm>    - bakke        (ex: b 200)")
    print(" l <grader>- drej venstre (ex: l 90)")
    print(" r <grader>- drej hojre   (ex: r 90)")
    print(" q         - afslut")
    print("============================")

menu()

while True:
    try:
        raw = raw_input("kommando: ").strip().lower()
    except NameError:
        raw = input("kommando: ").strip().lower()

    if not raw:
        continue

    parts = raw.split()
    cmd   = parts[0]

    if cmd == "q":
        print("Afslutter")
        break

    elif cmd == "d" and len(parts) == 2:
        mm = float(parts[1])
        print("Koer {} mm frem...".format(mm))
        drive(mm)
        print("Faerdig. Maal om afstanden passer.")
        print("Hvis for kort: oeg WHEEL_DIAM_MM")
        print("Hvis for langt: reducer WHEEL_DIAM_MM")

    elif cmd == "b" and len(parts) == 2:
        mm = float(parts[1])
        print("Bakker {} mm...".format(mm))
        drive(-mm)
        print("Faerdig.")

    elif cmd == "l" and len(parts) == 2:
        deg = float(parts[1])
        print("Drejer {} grader til venstre...".format(deg))
        turn(deg)
        print("Faerdig. Maal om vinklen passer.")
        print("Hvis for lidt: oeg WHEEL_BASE_MM")
        print("Hvis for meget: reducer WHEEL_BASE_MM")

    elif cmd == "r" and len(parts) == 2:
        deg = float(parts[1])
        print("Drejer {} grader til hojre...".format(deg))
        turn(-deg)
        print("Faerdig.")

    else:
        print("Ukendt kommando - se menu ovenfor")
        menu()