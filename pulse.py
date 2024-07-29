import time

import RPi.GPIO as GPIO

TRIGGER_PIN = 4

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIGGER_PIN, GPIO.OUT)


def generate_trigger_pulse(continued=False):
    GPIO.output(TRIGGER_PIN, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(TRIGGER_PIN, GPIO.LOW)
    time.sleep(0.1 if continued else 0.01)


def clean():
    GPIO.output(TRIGGER_PIN, GPIO.LOW)
    GPIO.cleanup()
