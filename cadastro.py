import cv2
import os
import time
import names_list

cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

names = names_list.names
username = input('Username:  ')
while username in names:
    print("Username já existe! Por favor, escolha outro")
    username = input('Username:  ')
names.append(username)

if os.path.exists('names_list.py'):
        os.remove('names_list.py')
with open('names_list.py','w') as fout :
    fout.write(f"names={names}")

directory=f"users/{username}"
if not os.path.exists(directory):
    os.makedirs(directory)
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
   # img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        cv2.imwrite(f"{directory}/{username}_{str(count)}.jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
        time.sleep(1)

    if count >= 20: 
         print("Cadastro concluído!")
         break

    cv2.imshow("Test", img)
    cv2.waitKey(1)

# Do a bit of cleanup
cam.release()
cv2.destroyAllWindows()


