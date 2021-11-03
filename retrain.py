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

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

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

#newGFile = open("gestureNames24classes.txt", 'r')
newGFile = open("gestureNames21classes.txt", 'r')
newGClassNames = newGFile.read().split('\n')
numOfGClasses = len(newGClassNames)
#print(newGClassNames)
#print(numOfGClasses)

# load data
dataFilePath = r"./data_handLms_21classes_train.txt"
inputFile = open(dataFilePath, "r")

readLines = inputFile.readlines()
inputFile.close()

xsTrain = []
ysTrain = []

for readLine in readLines:
    dInstance = json.JSONDecoder().decode(readLine)
    #print(dInstance)
    handLandmarks = dInstance["landmarks"]
    gestureClass = dInstance["class"]
    #print(gestureClass)
    gClassId = newGClassNames.index(gestureClass)
    gClassProbList = [0] * numOfGClasses
    gClassProbList[gClassId] = 1
    #print("id: ", gClassId)
    xsTrain.append( [handLandmarks] )
    ysTrain.append(gClassProbList)
    


# Training tracking
#log_dir = os.path.join('Logs')
#tb_callback = TensorBoard(log_dir=log_dir)

# Load the gesture recognizer model
model = load_model("mp_hand_gesture")

# Initiate a new model
newModel = Sequential()

# Copying all the layers from the old model, except the output layers

newModel.add(model.layers[0]) # just copy the first layer (input)

# if necessary, prevent the already trained layers from being trained again 
# for layer in newModel.layers:
    #layer.trainable = False

# adding the new layers, including new output layers
# note: the "name" parameter is important and should be unique here to avoid error
newModel.add(Dense(64, name='newDenseLayer1', activation='relu'))
newModel.add(Dense(128, name='newDenseLayer2', activation='relu'))
newModel.add(Dense(256, name='newDenseLayer3', activation='relu'))
newModel.add(Dense(512, name='newDenseLayer4', activation='relu'))
newModel.add(Dense(256, name='newDenseLayer5', activation='relu'))
newModel.add(Dense(128, name='newDenseLayer6', activation='relu'))
newModel.add(Dense(64, name='newDenseLayer7', activation='relu'))
newModel.add(Dense(numOfGClasses, name='newDenseLayerOutput', activation='softmax'))

# compile the new Model
newModel.compile(optimizer='Adam',loss='categorical_crossentropy', metrics = ['categorical_accuracy'])

# train
newModel.fit(x=xsTrain, y=ysTrain, epochs=22)

# save it
newModel.save("newMPHandGestureModel_21classes")







