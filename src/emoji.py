import tensorflow as tf
import threading
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import MaxPooling2D
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import numpy as np
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.disable_v1_behavior()


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
emotion_model.load_weights('src/model.h5')
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

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]
global frame_number


def show_subject():
    # /home/msh/Downloads, "data/examples/vid.mp4"
    cap1 = cv2.VideoCapture("data/examples/vid.mp4")
    if not cap1.isOpened():
        print("Can't open the camera")
    global frame_number
    length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number += 2
    if frame_number >= length:
        exit()
    cap1.set(1, frame_number)
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1, (600, 500))
    bounding_box = cv2.CascadeClassifier(
        "C:\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=8)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0] = maxindex
    if flag1 is None:
        print("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        root.update()
        lmain.after(10, show_subject)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()


def show_avatar():
    # global show_text, imgtk2
    frame2 = cv2.imread(emoji_dict[show_text[0]])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(pic2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain3.configure(
        text=emotion_dict[show_text[0]], font=('Arial', 30, 'bold'))
    lmain2.configure(image=imgtk2)
    root.update()
    lmain2.after(10, show_avatar)


##
if __name__ == '__main__':
    frame_number = 0
    root = tk.Tk()
    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)

    lmain3 = tk.Label(master=root, bd=10, fg="#FFFFFF", bg='black')
    # main video playback
    lmain.pack(side=LEFT)
    lmain.place(x=50, y=250)
    # 3-> avatar TEXT
    lmain3.pack()
    lmain3.place(x=980, y=250)
    # 2-> avatar display
    lmain2.pack(side=RIGHT)
    lmain2.place(x=860, y=250)
    # window
    root.title("Photo To Avatar")
    root.geometry("1400x1000+100+10")
    root['bg'] = 'black'
    exitButton = Button(root, text='Close', fg="red", font=(
        'Arial', 20, 'bold'), command=root.destroy).pack(side=BOTTOM)
    threading.Thread(target=show_subject).start()
    threading.Thread(target=show_avatar).start()
    root.mainloop()
