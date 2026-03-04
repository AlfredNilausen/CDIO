from ev3dev2.motor import MediumMotor, OUTPUT_A
from ev3dev2.sensor import INPUT_1
from ev3dev2.sensor.lego import InfraredSensor
from time import sleep

# Medium motor i port A
motor = MediumMotor(OUTPUT_A)

# IR sensor i port 1
ir = InfraredSensor(INPUT_1)

print("IR kontrol startet")

while True:

    # Øverste venstre knap = frem
    if ir.top_left(channel=1):
        motor.on(40)

    # Nederste venstre knap = baglæns
    elif ir.bottom_left(channel=1):
        motor.on(-40)

    # Ellers stop
    else:
        motor.off()

    sleep(0.1)