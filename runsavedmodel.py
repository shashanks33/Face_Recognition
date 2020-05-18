import base64
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np 
from keras.preprocessing import image

model = load_model('./facerecognition_first_model.h5')

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Face recognition

url = 'http://192.168.43.158:4747'
cap = cv2.VideoCapture(url+"/video")

while True:
    _, frame = cap.read()
    face = face_extractor(frame)

    if type(face) is np.ndarray:
        face = cv2.resize(face, (224,224))
        img = Image.fromarray(face, 'RGB')

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        print(prediction)

        name = 'None matching'

        if(prediction[0][0] > 0.7):
            name = 'Dad'
        elif(prediction[0][1] > 0.7):
            name = 'Mom'
        elif(prediction[0][2] > 0.7):
            name = 'Shashank'
        cv2.putText(frame, name, (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face found", (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
