# University of North Texas
# Fall 2021
# Team project for class CSCE 5280 by Professor Mark Albert

# Team members:
#Solomon Ubani ( solomonubani@my.unt.edu )
#Sulav Poudyal ( sulav697@gmail.com )
#Yen Pham ( yenpham@my.unt.edu )
#Khoa Ho ( khoaho@my.unt.edu ) 
#Stephanie Brooks( StephanieBrooks2@my.unt.edu )

# module to train a classifer-model -> current choice: LSTM

# import libraries/packages
import os
import time
import math
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as pyplt
#import pandas

from sklearn import metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard

#from google.colab.patches import cv2_imshow
#from keras.preprocessing import image
#from keras.utils import np_utils

import threading
import json


# initialize time mark
cTime = 0
pTime = 0

# initialize mediapipe
mpDrawing = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.7)

# mpFaceDetection = mp.solutions.face_detection
# faceDetection = mpFaceDetection.FaceDetection(model_selection=0, min_detection_confidence=0.8)
## model_selection: 0 -> within 2 meters from camera, 1 -> 2-5 meters from camera

#mpFaceMesh = mp.solutions.face_mesh
#faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.8, min_tracking_confidence=0.8, static_image_mode=False)

# Load class names


newGFile = open("gestureNames21classes.txt", 'r')
newGClassNames = newGFile.read().split('\n')
numOfGClasses = len(newGClassNames)
#print(newGClassNames)
#print(numOfGClasses)
newGClassNames.append("unknown")
print("num of classes: ", numOfGClasses)

# load data
dataFilePath = r"./data_handLms_21classes_test.txt"
inputFile = open(dataFilePath, "r")

readLines = inputFile.readlines()
inputFile.close()

xsTest = []
ysTest = []

for readLine in readLines:
    dInstance = json.JSONDecoder().decode(readLine)
    #print(dInstance)
    handLandmarks = dInstance["landmarks"]
    gestureClass = dInstance["class"]
    #print(gestureClass)
    gClassId = newGClassNames.index(gestureClass)
    #gClassProbList = [0] * numOfGClasses
    #gClassProbList[gClassId] = 1
    #print("id: ", gClassId)
    xsTest.append( [handLandmarks] )
    ysTest.append(gClassId)
    
    

# Training tracking
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Load the gesture recognizer model
model = load_model("newMPHandGestureModel_21classes")

ysTest_pred_class_adjusted = []
ysTest_pred_class_unadjusted = []

ysTest_pred = model.predict(xsTest, callbacks=[tb_callback])

for yTest_pred in ysTest_pred:
    #yTest_pred is in form of [ [ 21x float ] ] -> list of lists even if only 1 instance of x was supplied
    predClassId = np.argmax(yTest_pred)
    #print(predClassId)  
    ysTest_pred_class_unadjusted.append(predClassId)

    if yTest_pred[predClassId] > 0.7 :
        # 1 class > 70% prob
        predClassName = newGClassNames[predClassId]  
    else:
        # unsure class
        predClassName = newGClassNames[-1]
        predClassId = len(newGClassNames)-1 # class = "unknown" 

    ysTest_pred_class_adjusted.append(predClassId)

#print("len xsTest: ", len(xsTest))
#print("len ysTest: ", len(ysTest))
#print("ysTest: ", ysTest)
#print("len ysTestPredClass: ", len(ysTest_pred_class_adjusted))
#print("ysTestPredClass: ", ysTest_pred_class_adjusted)
#print("each ysTestPredClass: ", ysTest_pred_class_adjusted[0])

#print("len ysTestPred: ", len(ysTest_pred))
#print("len each yTestPred: ", len(ysTest_pred[0]))
#print("each yTestPred: ", ysTest_pred[0])




cMatrix_u = [ [0 for col in range( numOfGClasses+1 ) ] for row in range( numOfGClasses+1 ) ]
cMatrix_a = [ [0 for col in range( numOfGClasses+1 ) ] for row in range( numOfGClasses+1 ) ]
#print(np.shape(cMatrix_u))


for idx in range(len(ysTest)):
    cMatrix_u[ ysTest[idx] ][ ysTest_pred_class_unadjusted[idx] ] += 1
    cMatrix_a[ ysTest[idx] ][ ysTest_pred_class_adjusted[idx] ] += 1

#confusionMatrix_test_unadjusted = metrics.confusion_matrix(ysTest, ysTest_pred_class_unadjusted)
#confusionMatrix_test_adjusted = metrics.confusion_matrix(ysTest, ysTest_pred_class_adjusted)

accuracy_test_unadjusted = metrics.accuracy_score(ysTest, ysTest_pred_class_unadjusted)
accuracy_test_adjusted = metrics.accuracy_score(ysTest, ysTest_pred_class_adjusted)

balancedAccuracy_test_unadjusted = metrics.balanced_accuracy_score(ysTest, ysTest_pred_class_unadjusted)
balancedAccuracy_test_adjusted = metrics.balanced_accuracy_score(ysTest, ysTest_pred_class_adjusted)

print("CM_u: ")
for row in cMatrix_u:
    print(row)
#print(confusionMatrix_test_unadjusted)
print("CM_a: ")
for row in cMatrix_a:
    print(row)
#print(confusionMatrix_test_adjusted)

print("Acc_u: ", accuracy_test_unadjusted)
print("Acc_a: ", accuracy_test_adjusted)

print("Bal Acc_u: ", balancedAccuracy_test_unadjusted)
print("Bal Acc_a: ", balancedAccuracy_test_adjusted)
