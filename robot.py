#!/usr/bin/env python3

from ev3dev2.motor import LargeMotor, OUTPUT_B, OUTPUT_C, SpeedPercent
import sys
import tty
import termios

# Opret motorer
left_motor = LargeMotor(OUTPUT_B)
right_motor = LargeMotor(OUTPUT_C)

def get_key():
    """Læser én tast uden at trykke Enter"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

print("Styr robotten med WASD. Tryk Q for at afslutte.")

while True:
    key = get_key()

    if key.lower() == 'w':
        left_motor.on(SpeedPercent(30))
        right_motor.on(SpeedPercent(30))

    elif key.lower() == 's':
        left_motor.on(SpeedPercent(-30))
        right_motor.on(SpeedPercent(-30))

    elif key.lower() == 'a':
        left_motor.on(SpeedPercent(-20))
        right_motor.on(SpeedPercent(20))

    elif key.lower() == 'd':
        left_motor.on(SpeedPercent(20))
        right_motor.on(SpeedPercent(-20))

    elif key.lower() == 'q':
        break

    else:
        left_motor.off()
        right_motor.off()

# Stop motorer når programmet afsluttes
left_motor.off()
right_motor.off()
print("Program afsluttet.")