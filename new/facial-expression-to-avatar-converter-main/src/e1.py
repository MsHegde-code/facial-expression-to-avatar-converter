import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import threading

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(
    3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('/home/msh/Downloads/facial-expression-to-avatar-converter-main/model.h5')
cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "angry",
                1: "happy",
                2: "neutral",
                3: "sad",
                4: "surprised",
                5: "disgusted",
                6: "fearful"}

cur_path = os.path.dirname(os.path.abspath(__file__))

emoji_dict = {
    0: cur_path+"/emojis/angry.png",
    1: cur_path+"/emojis/happy.png",
    2: cur_path+"/emojis/neutral.png",
    3: cur_path+"/emojis/sad.png",
    4: cur_path+"/emojis/surprised.png",
    5: cur_path+"/emojis/disgusted.png",
    6: cur_path+"/emojis/fearful.png"
}

def show_subject():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open the camera")
        return
    
    global frame_number
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to retrieve frame")
            break
        
        frame = cv2.resize(frame, (640, 500))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=8)
        
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            show_text[0] = maxindex
        
        # Display frame
        cv2.imshow('Subject View', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def show_avatar():
    while True:
        frame = cv2.imread(emoji_dict[show_text[0]])
        cv2.imshow('Avatar', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    show_text = [0]
    threading.Thread(target=show_subject).start()
    threading.Thread(target=show_avatar).start()

