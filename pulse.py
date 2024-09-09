import time

import RPi.GPIO as GPIO

TRIGGER_PIN = 4

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIGGER_PIN, GPIO.OUT)


def generate_trigger_pulse():
    GPIO.output(TRIGGER_PIN, GPIO.HIGH)
    time.sleep(0.001)
    GPIO.output(TRIGGER_PIN, GPIO.LOW)


def clean():
    GPIO.output(TRIGGER_PIN, GPIO.LOW)
    GPIO.cleanup()
