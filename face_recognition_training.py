import cv2
import numpy as np
from PIL import Image
import os
import names_list

# Path for face image database
names = names_list.names

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def imgs_ids_train(names):
    totalSamples=[]
    total_ids = []
    faceSamples=[]
    ids = []
    id = 0
    for name in names:
        path = f'users/{name}'
        print(path)
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        print(imagePaths)
        

        for imagePath in imagePaths:
            print(imagePath)
            PIL_img = Image.open(imagePath).convert('L') 
            img_numpy = np.array(PIL_img,'uint8')

            name = path.split("/")[1]
            print(f"---------{name} {id}")
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        id+=1
        total_ids.append(ids)
        totalSamples.append(faceSamples)

    return faceSamples,ids

faces,ids = imgs_ids_train(names)
print(ids)
print(faces)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer.yml')

print("{0} faces detected".format(len(np.unique(names))))
