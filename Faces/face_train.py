import os
import cv2 as cv
import numpy as np

"""
This demonstrates two methods to create a list of people's names (labels) for face recognition:

1. Manual way:
   - Define a list `people` with names of individuals.
   - Useful for small or known datasets.

2. Automated way:
   - Reads the folder names inside the training dataset directory.
   - Each folder corresponds to a person.
   - Automatically generates the list `p` by scanning the folder names.

The training folder path should contain subfolders named after each person, e.g.:
'train/Ben Afflek/', 'train/Elton John/', etc.

The resulting list can be used as labels for training face recognition models.
"""

# Manual list of people
people = ['Ben Afflek', 'ELton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# OR automated way to list folder names inside the train directory
# p = []
# train_dir = r'\\wsl.localhost\Ubuntu-20.04\home\ppallai\Programs\Libraries\Opencv\Resources\Faces\train'

# for folder_name in os.listdir(train_dir):
#     p.append(folder_name)
    
# print(p)  # Prints the list of people/folder names found in the training directory

# Path to training data folder
train_dir = r'\\wsl.localhost\Ubuntu-20.04\home\ppallai\Programs\Libraries\Opencv\Resources\Faces\train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

# Dynamically get people list from folder names to avoid case errors
people = os.listdir(train_dir)
print(f"People found: {people}")

def create_train():
    for person in people:
        path = os.path.join(train_dir, person)
        label = people.index(person)
        
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            
            for(x, y, w, h) in face_rect:
                face_roi = gray[y:y+h, x:x+w]
                features.append(face_roi)
                labels.append(label)
                
create_train()
print('Training done................')

print(f'Length of the features = {len(features)}')
print(f'Length of the labels = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the reconginer on the features list and the labels list 

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

cv.waitKey(0)
cv.destroyAllWindows()
