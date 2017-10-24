import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import serial
import serial.tools.list_ports
from PyQt5.QtGui import *
import PyQt5.QtWidgets
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtWidgets import *

ard = serial.Serial()
ard.baudrate = 115200



def collectData():
    #ser = Serial.serial
    pass

def get_indicators(vec):
    '''
    Source: https://github.com/VikParuchuri/simpsons-scripts
    '''
    mean = np.mean(vec)
    slope = calc_slope(np.arange(len(vec)),vec)
    std = np.std(vec)
    mx = np.max(vec)
    mi = np.min(vec)
    sdev = np.std(vec)
    return mean, slope, std, mx, mi, sdev

def calc_slope(x,y):
    '''
    Source: https://github.com/VikParuchuri/simpsons-scripts
    '''
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_dev = np.sum(np.abs(np.subtract(x,x_mean)))
    y_dev = np.sum(np.abs(np.subtract(y,y_mean)))

    slope = (x_dev*y_dev)/(x_dev*x_dev)
    return slope

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

def printCoilDataTagged():
    try:
        connectArduino()
        while True:
            reading = list(ard.readline())
            print(reading)
            tag = reading[0:4]
            value = reading[5:]
            print('"%s": "%s"' % (tag, reading))
            
    except Exception as exc:
        print('Exception: "%s"' % exc)

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

class TrackingGui(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        xlabel = QLabel('X:', self) 
        xlabel.setMaximumWidth(40)
        ylabel = QLabel('Y:', self) 
        ylabel.setMaximumWidth(40)
        zlabel = QLabel('Z:', self)
        zlabel.setMaximumWidth(40)

        self.xline = QLineEdit('')
        self.xline.setReadOnly(True)
        self.yline = QLineEdit('')
        self.yline.setReadOnly(True)
        self.zline = QLineEdit('')
        self.zline.setReadOnly(True)
        
        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(xlabel, 0, 0)
        grid.addWidget(self.xline, 0, 1)

        grid.addWidget(ylabel, 1, 0)
        grid.addWidget(self.yline, 1, 1)
        
        grid.addWidget(zlabel, 2, 0)
        grid.addWidget(self.zline, 2, 1)

        grid.columnStretch(0)

        vbox = QVBoxLayout()
        vbox.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addLayout(grid, 0)
        hbox.addLayout(vbox, 1)

        self.setLayout(hbox)
        self.setGeometry(1000, 700, 1000, 700)
        self.setWindowTitle('3D Tracking GUI')
        self.center()
        self.show()
        
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
            
if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        ex = TrackingGui()
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
