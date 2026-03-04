#!/usr/bin/env python3

from ev3dev2.motor import LargeMotor, OUTPUT_B, OUTPUT_C, SpeedPercent
from time import sleep

# Opret motorer
left_motor = LargeMotor(OUTPUT_B)
right_motor = LargeMotor(OUTPUT_C)

# Kør frem
left_motor.on(SpeedPercent(30))
right_motor.on(SpeedPercent(30))
sleep(2)

# Stop
left_motor.off()
right_motor.off()

# Drej
left_motor.on(SpeedPercent(30))
right_motor.on(SpeedPercent(-30))
sleep(1)

# Stop igen
left_motor.off()
right_motor.off()