# University of North Texas
# Fall 2021
# Team project for class CSCE 5280 by Professor Mark Albert

# Team members:
#Solomon Ubani ( solomonubani@my.unt.edu )
#Sulav Poudyal ( sulav697@gmail.com )
#Yen Pham ( yenpham@my.unt.edu )
#Khoa Ho ( khoaho@my.unt.edu ) 
#Stephanie Brooks( StephanieBrooks2@my.unt.edu )

# module to extract key points from a video

# -*- coding: utf-8 -*-
#"""VGR_MP.ipynb

#Automatically generated by Colaboratory.

#Original file is located at
#    https://colab.research.google.com/drive/1YTtC5gLzRdV5gNlQvgPDzwhMPzLrjdnE
#"""

#!pip install mediapipe

# Computer Vision features
import cv2
from google.colab.patches import cv2_imshow
from keras.preprocessing import image
from keras.utils import np_utils
import mediapipe as mp

# Data processing
import math
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt

# File path & temporal processes
import os
import time

# Modeling
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Set the holistic model
mp_holistic = mp.solutions.holistic

# Set the drawing utilities
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    
    # Converts the image color.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    # Prevents image write
    image.flags.writeable = False                  
    
    # Makes the prediction.
    results = model.process(image)
    
    # Enables image write.
    image.flags.writeable = True
    
    # Convert back to BGR.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    
    # Return the image and prediction results.
    return image,results

# draw_landmarks: Takes a frame and results then applies the landmark 
# visualizations to hand and pose.
def draw_landmarks(image, results):
    
    # Draw left hand points.                    
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, 
                              mp_holistic.HAND_CONNECTIONS)
    # Draw right hand points.
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, 
                              mp_holistic.HAND_CONNECTIONS) 
    
    # Draw pose points.
    mp_drawing.draw_landmarks(image, results.pose_landmarks, 
                              mp_holistic.POSE_CONNECTIONS)

# extract_keypoints: Gets the x,y,z coordinates of the keypoints of a frame and returns a concatenated
# array of those coordinates for the pose, left, and right hand.
def extract_keypoints(results):

    # Gets and flattens the coordinates for each of the landmark areas. If there
    # are no values for the frame, 0's are returned.
    pose = np.array([[res.x, res.y, res.z, res.visibility] for 
                     res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for 
                   res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for 
                   res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Returns the concatenated np array for each of the landmarks.
    return np.concatenate([pose, lh, rh])

# Set the folder path for the numpy arrays.
DATA_PATH = os.path.join('/content/Data')


"""Takes the video, performs keypoints extractions by frame, and saves the resulting numpy arrays to a folder. **** The numpy_array_path needs some tweaking****"""

# extractKPFromVid: Performs keypoints extraction on each frame of the input video
# and saves the keypoints to a numpy array folder
def extractKPFromVid(videoMP4,action):
  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # START
    print("START KP extraction from vid: \n")
    # Sets count.
    count = 0
    
    # Set the numpyList
    numpyList = []

    # Sets the videoFile name. **INCLUDE** the extension
    videoFile = videoMP4

    # Captures the video
    cap = cv2.VideoCapture(videoMP4)

    # Set the framerate 
    frameRate = cap.get(5)

    print("framerate: ",frameRate,"\n")

    # While the video is running, read in the video frames
    # and extract the keypoints to a file. 
    while(cap.isOpened()):
      print("Start video processing. . .")
      # Sets the frame number
      frameId = cap.get(1)

      # Reads in the frame.
      ret, frame = cap.read()
      print(ret)
      # Display the video frame with landmarks overlaid.
      #cv2_imshow(frame)

      # If there are no more frames, the capturing stops.
      # Otherwise, the next frame is read in.
      if (ret != True):
        print("End of video: ",videoMP4)
        break

      # ---- Extract and append the keypoints----.
      # Keypoints detections.
      image,results = mediapipe_detection(frame,holistic)

      # Increment count
      count+=1

      #---Export keypoints---
      # Get the keypoints array
      keypoints = extract_keypoints(results)

      # Append the keypoints to the numpyList.
      if(count<89):
        numpyList.append(keypoints)     
        break

    # Stops the capture.
    cap.release()
    cv2.destroyAllWindows

    # Output finish message.
    print ("Frame Capture Finished.")

    # Return the list of numpy arrays.
    return numpyList

# Set the folder path for the numpy arrays.
DATA_PATH = os.path.join('/content/Data')

# Read in the text file to a dataframe.
#df_gestures = pd.read_csv("/content/Data/labelmap.txt",sep=",",header=0)

"""Video keypoint Extraction"""

##-------------------##
# Set the gestures, window, and sequences.
gestures = ['Down','Left','Right','Up']
window,sequences = [],[]
 
# Creates the map for gesture to classification values.
gestureLabelMap = {label:num for num,label in enumerate(gestures)}

# Loops through each video per gesture, processes the video keypoints,
#  and builds a list of concatenated numpy arrays.
for gesture in gestures:

  for root,dirs,files in os.walk(os.path.join(DATA_PATH,gesture)):
    print(os.path.join(DATA_PATH,gesture))
    for fil in files:
      # Extract frames and set the list of numpy arrays from the video.
      window = extractKPFromVid(os.path.join(DATA_PATH,gesture,fil),gesture)

  # Append the numpyLists for the gesture.
  sequences.append(window)
  print("Sequence complete for: ",gesture,"\nSEQUENCE: ",sequences)

# Output label dimensions.
np.array(labels).shape

# Set X to list of numpy arrays.
X = np.array(sequences)

# Output X dimensions.
X.shape

# Set the y value
y=to_categorical(labels).astype(int)

"""Train Test Split"""

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.05)

"""Model Training"""

# Training tracking
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Set the model
model = Sequential()

# Add the LSTM layers. 
# **Note: input_shape: If X.shape values were [a,b,c] it would just be b,c.
# If X.shape outputs (4,30,258), the input_shape is (30,258)**
model.add(LSTM(64,return_sequences=True,activation='relu',input_shape=(30,258)))
model.add(LSTM(128,return_sequences=True,activation='relu'))
model.add(LSTM(64,return_sequences=False,activation='relu'))
# Add the Dense layers
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(gestures.shape[0],activation='softmax'))

# Compile the model
# multiclass classification model --> categorical cross entropy used.
model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics = ['categorical_accuracy'])

# Fit the model
model.fit(X_train,y_train,epochs=1000,callbacks=[tb_callback])
