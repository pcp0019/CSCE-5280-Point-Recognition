# University of North Texas
# Fall 2021
# Team project for class CSCE 5280 by Professor Mark Albert

# Team members:
#Solomon Ubani ( solomonubani@my.unt.edu )
#Sulav Poudyal ( sulav697@gmail.com )
#Yen Pham ( yenpham@my.unt.edu )
#Khoa Ho ( khoaho@my.unt.edu ) 
#Stephanie Brooks( StephanieBrooks2@my.unt.edu )

# module to extract data (key-points or landmarks) from raw data (images or frame)

# import libraries/packages
import os
import time

import numpy as np
import cv2
import mediapipe as mp

import glob
import json


# initialize mediapipe
mpDrawing = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.1, min_tracking_confidence=0.1)

#os.chdir('')

dataDirectory = r"../DataDir/dataset_test/"
dataFilePath = r"./data_handLms_21classes_test.txt"
# Load class names
gClassNames = [directory for directory in os.listdir(dataDirectory) if os.path.isdir(dataDirectory + directory)]
numOfGClasses = len(gClassNames)
print("G Classes: ", gClassNames)
print("Num of classes: ", numOfGClasses)

outputFile = open(dataFilePath, "w")

fileCount = 0
recogCount = 0
for gClassName in gClassNames:
    gDataDir = dataDirectory + gClassName
    print(gDataDir)
    for dFile in glob.glob(gDataDir + r"/*"):
        frame = cv2.imread(dFile)
        x , y, c = frame.shape 
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        handResult = hands.process(framergb)

        fileCount += 1
        if handResult.multi_hand_landmarks:
            # print("hand result: ", handResult.multi_hand_landmarks)
            # -- Khoa, Oct 5 2021.
            # hand key point extractor model default to 1 hand, so we are not using multiLandmarks here 
            # we still leave the related code in, in case we want to change the model to take multi hands
            # multiLandmarks = []
            # landmarks = []
            for handslms in handResult.multi_hand_landmarks:
                # print(handslms)
                landmarks = []
                for lm in handslms.landmark:
                    # print(lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    # lmz = int(lm.z)
                    landmarks.append([lmx, lmy])
                # end inner for-loop

                # multiLandmarks.append(landmarks)

            # end outer for-loop

            #if len(multiLandmarks > 1):
                #landmarks = multiLandmarks[0]

            outputStr = json.JSONEncoder().encode({"class": gClassName, "landmarks":landmarks})
            #print(outputStr)

            outputFile.write(outputStr)
            outputFile.write("\n")

            recogCount += 1

    print("Running Sum file count: ", fileCount)
    print("Running Sum recog count: ", recogCount)  
    print("____________________________________\n")

outputFile.close()

print("Total file count: ", fileCount)
print("Total recog count: ", recogCount)
#inputFile = open(dataFilePath, "r")

#readLines = inputFile.readlines()
#inputFile.close()

#for readLine in readLines:
    #readStrs = readLine.split(",")
    #print("read: ", readStrs)



    