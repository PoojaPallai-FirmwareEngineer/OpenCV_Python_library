import os
import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Manual list of people
people = ['Ben Afflek', 'ELton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# features = np.load('Features.npy')
# labels = np.load('Labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# img = cv.imread(r'\\wsl.localhost\Ubuntu-20.04\home\ppallai\Programs\Libraries\Opencv\Resources\Faces\val\ben_afflekResources/Faces/val/ben_afflek/2.jpg')

img_path = r'\\wsl.localhost\Ubuntu-20.04\home\ppallai\Programs\Libraries\Opencv\Resources\Faces\val\ben_afflek\2.jpg'
img = cv.imread(img_path)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Person", gray)

# Detect the face in th image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]
    
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)
cv.destroyAllWindows()
