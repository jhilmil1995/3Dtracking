import os
import time
from datetime import datetime
import sys
import random
import uuid
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

def standarizeData():
    X = []
    y = []
    try:
        os.chdir(os.getcwd()+'\\data')
        for path, subdir, files in os.walk('.'):
            for directory in subdir:            
                for file in os.listdir(os.getcwd() + '\\'+directory):
                    #directory = list(map(int,directory.split('_')[1:]))
                    y.append(directory)
                    #y.append(list(map(int,directory.split('_')[1:])))
                    #print(list(map(int,directory.split('_')[1:])))
                    filename = os.getcwd()+ '\\'+directory + '\\' + file
                    f = open(filename, 'r')
                    data = f.read()#[1:-1]
                    data = data[1:-1]
                    data = [float(s) for s in data.split(',')]
                    X.append(data)
                    f.close()
    except Exception as exc:
        print(exc)
    #print(y)
    #print(X)
    os.chdir(os.getcwd()+'\\..')
    dataFile = 'features.py'
    fileObj = open(dataFile, 'w')
    fileObj.write('X = ' + pprint.pformat(X)+ '\n')
    fileObj.write('y = ' + pprint.pformat(y)+ '\n')
    fileObj.close()
    #print(len(X), len(y))

    
def testRegression():
    from sklearn.cross_validation import train_test_split
    from sklearn import linear_model
    
    import features
    X = np.array(features.X)
    y = np.array(features.y)

    testSize = .2
    randomState=1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState) 
    
    ols = linear_model.LinearRegression(copy_X=True)
    ols.fit(X_train, y_train)

    results = ols.predict(X_test)
    #print(type(results))
    print([round(elem) for elem in results])
    
    """estimatorName=['Linear Regression']
    predictions=[]
    
    predictions.append(ols.predict(X_test))
    predictions.append(clf1.predict(X_test))
    predictions.append(clf2.predict(X_test))
    predictions.append(clf3.predict(X_test))
    predictions.append(eclf.predict(X_test))"""
    """
    for i in range(1):

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
    
    """

def trainClassifier():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, SVR
    from sklearn.ensemble import VotingClassifier
    from itertools import product
    from sklearn.cross_validation import train_test_split
    #from sklearn.ensemble import RandomForestClassifier

    #This is whatever you saved your X and y data into
    import features

    X = np.array(features.X)
    y = np.array(features.y)

    testSize = .3
    randomState=1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)

    #Classification
    clf0 = svm.SVC(kernel='linear')
    clf1 = DecisionTreeClassifier(max_depth=4)
    clf2 = KNeighborsClassifier(n_neighbors=7)
    clf3 = SVC(kernel='rbf', probability=True)
    eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3)],
                        voting='soft', weights=[2, 1, 2])

    #Regression
    svrRbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    scrLin = SVR(kernel='linear', C=1e3)
    svrPoly = SVR(kernel='poly', C=1e3, degree=2)
    
    
    clf0.fit(X_train,y_train)
    clf1.fit(X_train,y_train)
    clf2.fit(X_train,y_train)
    clf3.fit(X_train,y_train)
    eclf.fit(X_train,y_train)

    svrRbf.fit(X_train,y_train)
    svrLin.fit(X_train,y_train)
    svrPoly.fit(X_train,y_train)
    
    estimatorName=['linear SVC', 'Decision Tree', 'K Neighbors',
                   'rbf Kernel SVC', 'Voting Classifier', 'SVR-RBF',
                   'SVR-Lin', 'SVR-Poly']
    predictions=[]
    predictions.append(clf0.predict(X_test))
    predictions.append(clf1.predict(X_test))
    predictions.append(clf2.predict(X_test))
    predictions.append(clf3.predict(X_test))
    predictions.append(eclf.predict(X_test))
    
    predictions.append(svrRbf.predict(X_test))
    predictions.append(svrLin.predict(X_test))
    predictions.append(svrPoly.predict(X_test))
    
    for i in range(len(predictions)):

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

    """forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=1)
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
    """


def collectData():
    path = 'C:\\Users\\Sara Srivastav\\Documents\\senior design\\3Dtracking\\python\\data'
    #ser = Serial.serial
    """
        #Coil1
        coil1 = np.random.uniform(low=3, high=5, size=(100,))
        coil2 = np.random.uniform(low=3, high=5, size=(100,))
        coil3 = np.random.uniform(low=0, high=3, size=(100,))
        coil4 = np.random.uniform(low=0, high=3, size=(100,))
        x = '1'
        y = '1'
        z = '0'
        #Coil2
        coil1 = np.random.uniform(low=3, high=5, size=(100,))
        coil2 = np.random.uniform(low=0, high=3, size=(100,))
        coil3 = np.random.uniform(low=3, high=5, size=(100,))
        coil4 = np.random.uniform(low=0, high=3, size=(100,))
        x = '1'
        y = '2'
        z = '0'
        #Coil3
        coil1 = np.random.uniform(low=0, high=3, size=(100,))
        coil2 = np.random.uniform(low=3, high=5, size=(100,))
        coil3 = np.random.uniform(low=0, high=3, size=(100,))
        coil4 = np.random.uniform(low=3, high=5, size=(100,))
        x = '2'
        y = '1'
        z = '0'
        #Coil4
        coil1 = np.random.uniform(low=0, high=3, size=(100,))
        coil2 = np.random.uniform(low=0, high=3, size=(100,))
        coil3 = np.random.uniform(low=3, high=5, size=(100,))
        coil4 = np.random.uniform(low=3, high=5, size=(100,))
        x = '2'
        y = '2'
        z = '0'
    """
    numsamples = 30
    for i in range(numsamples):
        #Coil1
        coil1 = np.random.uniform(low=3, high=5, size=(100,))
        coil2 = np.random.uniform(low=3, high=5, size=(100,))
        coil3 = np.random.uniform(low=0, high=3, size=(100,))
        coil4 = np.random.uniform(low=0, high=3, size=(100,))
        x = '1'
        y = '1'
        z = '0'
        
        dataFolder = path + "\\cord_%s_%s_%s" %  (x, y, z)
        if not os.path.exists(dataFolder):
            os.makedirs(dataFolder)
        dataFile = "%s.txt" %  (datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')[:-3])
        featuresC1 = get_indicators(coil1)
        featuresC2 = get_indicators(coil2)
        featuresC3 = get_indicators(coil3)
        featuresC4 = get_indicators(coil4)
        features = []
        features.extend(featuresC1)
        features.extend(featuresC2)
        features.extend(featuresC3)
        features.extend(featuresC4)
        #print(features)
        f = open(dataFolder + "\\"+ dataFile, 'w')
        f.write(str(features))
        f.close()
    
    for i in range(numsamples):
        #Coil2
        coil1 = np.random.uniform(low=3, high=5, size=(100,))
        coil2 = np.random.uniform(low=0, high=3, size=(100,))
        coil3 = np.random.uniform(low=3, high=5, size=(100,))
        coil4 = np.random.uniform(low=0, high=3, size=(100,))
        x = '1'
        y = '2'
        z = '0'
        
        dataFolder = path + "\\cord_%s_%s_%s" %  (x, y, z)
        if not os.path.exists(dataFolder):
            os.makedirs(dataFolder)
        dataFile = "%s.txt" %  (datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')[:-3])
        featuresC1 = get_indicators(coil1)
        featuresC2 = get_indicators(coil2)
        featuresC3 = get_indicators(coil3)
        featuresC4 = get_indicators(coil4)
        features = []
        features.extend(featuresC1)
        features.extend(featuresC2)
        features.extend(featuresC3)
        features.extend(featuresC4)
        #print(features)
        f = open(dataFolder + "\\"+ dataFile, 'w')
        f.write(str(features))
        f.close()

    for i in range(numsamples):
        #Coil3
        coil1 = np.random.uniform(low=0, high=3, size=(100,))
        coil2 = np.random.uniform(low=3, high=5, size=(100,))
        coil3 = np.random.uniform(low=0, high=3, size=(100,))
        coil4 = np.random.uniform(low=3, high=5, size=(100,))
        x = '2'
        y = '1'
        z = '0'
        
        dataFolder = path + "\\cord_%s_%s_%s" %  (x, y, z)
        if not os.path.exists(dataFolder):
            os.makedirs(dataFolder)
        dataFile = "%s.txt" %  (datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')[:-3])
        featuresC1 = get_indicators(coil1)
        featuresC2 = get_indicators(coil2)
        featuresC3 = get_indicators(coil3)
        featuresC4 = get_indicators(coil4)
        features = []
        features.extend(featuresC1)
        features.extend(featuresC2)
        features.extend(featuresC3)
        features.extend(featuresC4)
        #print(features)
        f = open(dataFolder + "\\"+ dataFile, 'w')
        f.write(str(features))
        f.close()

    for i in range(numsamples):
        #Coil4
        coil1 = np.random.uniform(low=0, high=3, size=(100,))
        coil2 = np.random.uniform(low=0, high=3, size=(100,))
        coil3 = np.random.uniform(low=3, high=5, size=(100,))
        coil4 = np.random.uniform(low=3, high=5, size=(100,))
        x = '2'
        y = '2'
        z = '0'
        
        dataFolder = path + "\\cord_%s_%s_%s" %  (x, y, z)
        if not os.path.exists(dataFolder):
            os.makedirs(dataFolder)
        dataFile = "%s.txt" %  (datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')[:-3])
        featuresC1 = get_indicators(coil1)
        featuresC2 = get_indicators(coil2)
        featuresC3 = get_indicators(coil3)
        featuresC4 = get_indicators(coil4)
        features = []
        features.extend(featuresC1)
        features.extend(featuresC2)
        features.extend(featuresC3)
        features.extend(featuresC4)
        #print(features)
        f = open(dataFolder + "\\"+ dataFile, 'w')
        f.write(str(features))
        f.close()
        

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
            #stirng parser for foldername
            #output = list(map(int,directory.split('_')[1:]))
            #x = output[0]
            #y = output[1]
            #z = output[2]
            
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
        #collectData()
        #standarizeData()
        #testRegression()

        trainClassifier()
        #app = QApplication(sys.argv)
        #ex = TrackingGui()
        #sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
