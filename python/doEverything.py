import os
import time
import sys
import random
import pprint
import numpy as np
from sklearn import svm
import serial
import serial.tools.list_ports
from PyQt5.QtGui import *
import PyQt5.QtWidgets
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtWidgets import *

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ard = serial.Serial()
ard.baudrate = 115200

def trainClassifier():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier
    from itertools import product
    from sklearn.cross_validation import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    #This is whatever you saved your X and y data into
    import carAndJunkFeatures

    X = np.array(carAndJunkFeatures.X)
    y = np.array(carAndJunkFeatures.y)

    testSize = .3
    randomState=1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)

    clf0 = svm.SVC(kernel='linear')
    clf1 = DecisionTreeClassifier(max_depth=4)
    clf2 = KNeighborsClassifier(n_neighbors=7)
    clf3 = SVC(kernel='rbf', probability=True)
    eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3)],
                        voting='soft', weights=[2, 1, 2])
    
    clf0.fit(X_train,y_train)
    clf1.fit(X_train,y_train)
    clf2.fit(X_train,y_train)
    clf3.fit(X_train,y_train)
    eclf.fit(X_train,y_train)
    
    estimatorName=['linear SVC', 'Decision Tree', 'K Neighbors', 'rbf Kernel SVC', 'Voting Classifier']
    predictions=[]
    predictions.append(clf0.predict(X_test))
    predictions.append(clf1.predict(X_test))
    predictions.append(clf2.predict(X_test))
    predictions.append(clf3.predict(X_test))
    predictions.append(eclf.predict(X_test))
    
    for i in range(5):

        pprint.pprint(estimatorName[i])
        pprint.pprint('Predictions:'), pprint.pprint(np.array(predictions[i]))
        pprint.pprint('Ground Truth:'), pprint.pprint(np.array(y_test))
        
        predVsTruth=predictions[i]==y_test        
        pprint.pprint(predVsTruth)
        numCases =(len(predictions[i]))
        numTrue = np.sum(predVsTruth)
        numFalse = numCases - numTrue
        print('Accuracy is: "%s"' % (numTrue/numCases*100))
        print('Number True: "%s", Number False: "%s"\n\n' % (numTrue,numFalse))

    #Must download Graphviz.exe and pip install graphviz for this to work
    #Gives a tree representation of the decision tree decision parameters. 
    os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'
    dot_data = tree.export_graphviz(clf1, out_file=None) 
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_pdf("carClassifier.pdf") 

    #The graphing portion of this is not working 100% currently
    feat_labels = ['m','sf','mx','mi','sdev','amin','smin','stmin','apeak','speak','stpeak','acep','scep','stcep','aacep','sscep','stsscep','zcc','zccn','spread','skewness','savss','mavss']

    forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=1)
    forest.fit(X_train,y_train)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
                                feat_labels[indices[f]], 
                                importances[indices[f]]))

    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]), 
            importances[indices],
            color='lightblue', 
            align='center')
    for f in range(X_train.shape[1]):
        plt.xticks(range(X_train.shape[1]), 
                   feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    #plt.savefig('./random_forest.png', dpi=300)
    plt.show()
    


def collectData():
    path = 'C:\\Users\\Gersemi\\Desktop\\Github\\3Dtracking\\python'
    #ser = Serial.serial
    """for i in range(4):
        coil1 = np.random.uniform(low=0.5, high=5.0, size=(100,))
        coil

    """
    for i in range(5):
        coil1 = np.random.uniform(low=4, high=5, size=(100,))
        coil2 = np.random.uniform(low=2, high=4, size=(100,))
        
    ##    cordCopy = cord.copy()
        x = '1'
        y = '1'
        z = '0'
        dataFile = "cord_%s_%s_%s.py" %  (x, y, z)
        featuresC1 = get_indicators(coil1)
        featuresC2 = get_indicators(coil2)
        features = []
        features.extend(featuresC1)
        features.extend(featuresC2)
        print(features)
    ##    cordCopy = cordCopy.tolist()
    ##    cordCopy.append(features)
    ##    print(dataFile)
        #if os.
        f = open(dataFile, 'w')
        f.write('X = ' + pprint.pformat(features) + '\n')
        f.close()
        
        """cord-1-2-0 = np.random.uniform(low=1.1, high=2.0, size=(100,))
        dataFile-1-2-0 = '..\readings\cord-1-2-0.txt'
        
        cord-2-1-0 = np.random.uniform(low=3.1, high=4.0, size=(100,))
        dataFile-2-1-0 = '..\readings\cord-2-1-0.txt'
        
        cord-2-2-0 = np.random.uniform(low=2.1, high=3.0, size=(100,))
        dataFile-2-2-0 = '..\readings\cord-2-2-0.txt'"""
        

def get_indicators(vec):
    '''
    Source: https://github.com/VikParuchuri/simpsons-scripts
    '''
    mean = np.mean(vec)
    slope = calc_slope(np.arange(len(vec)),vec)
    #std = np.std(vec)
    mx = np.max(vec)
    mi = np.min(vec)
    sdev = np.std(vec)
    return [mean, slope, mx, mi, sdev]

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

        #Coordinate box setup
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
        
        #Plot setup
         # a figure instance to plot on
        self.figure = Figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        #self.fig = plt.figure()
        self.ax = self.figure.add_subplot(111, projection='3d')
        
        #Placing objects onto the GUI
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
        vbox.addWidget(self.canvas)
        vbox.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addLayout(grid, 0)
        hbox.addLayout(vbox, 1)

        self.setLayout(hbox)
        self.setGeometry(1000, 700, 1000, 700)
        self.setWindowTitle('3D Tracking GUI')
        self.center()
        self.show()

        timer = QTimer(self)
        timer.timeout.connect(self.updatePlot)
        timer.start(20)
        
    def updatePlot(self):
        try:
            ''' plot some random stuff '''
            #x, y, z = classifyData()
            x = random.uniform(0.0,30.0)
            y = random.uniform(0.0,20.0)
            z = random.uniform(0.0,10.0)
            
            self.xline.setText(str(round(x,2))+' mm')
            self.yline.setText(str(round(y,2))+' mm')
            self.zline.setText(str(round(z,2))+' mm')
            
            # create an axis
            #ax = self.figure.add_subplot(111, projection='3d')

            # discards the old graph
            self.ax.clear()

            # plot data
            self.ax.set_xlim(0, 30, 5)
            self.ax.set_ylim(0, 20, 5)
            self.ax.set_zlim(0, 10, 2)
            self.ax.scatter(x, y, z, 'gray')

            # refresh canvas
            self.canvas.draw()
        except Exception as exc:
            print('Exception: "%s"' % exc)
        
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
            
if __name__ == '__main__':
    
    try:
        try:
            connectArduino()
        except:
            print('No Arduino')
        collectData()
        #app = QApplication(sys.argv)
        #ex = TrackingGui()
        #sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
