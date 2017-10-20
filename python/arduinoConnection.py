import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import serial
import serial.tools.list_ports

ard = serial.Serial()
ard.baudrate = 115200

def collectData():
    #ser = Serial.serial
    pass

def connectArduino():
    if ard.is_open:
        ard.close()
    arduinoPorts = [
        p.device
        for p in serial.tools.list_ports.comports()
        if 'USB-SERIAL CH340' in p.description
    ]
    if not arduinoPorts:
        raise IOError("No Arduino found")
    if len(arduinoPorts) > 1:
        warnings.warn('Multiple Arduinos found - using the first')
    ard.port = arduinoPorts[0]
    ard.open()

def printCoilData():
    try:

        connectArduino()
        while True:
            for i in range(4):
                reading = ard.readline()
                coilReading = reading.replace('coil'+str(i), '')
                print('Coil"%s" reading: "%s"' % (i,float(coilReading)))

    except Exception as exc:
        print('Exception: "%s"' % exc)
