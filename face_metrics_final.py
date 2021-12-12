import mediapipe as mp
import time
import cv2
from fer import FER
import math
import winsound
import numpy as np
import os 
import names_list
from datetime import datetime
import keyboard as kd
import json
import sys

# Parâmetros para o Face Mesh
EYES_LIST = [159, 145]
HEAD_LIST = [6, 1]
PISCA_FLAG = False
PISCA_THRES = 0.17
n_pisc = 0
NUM_FACE = 2

# Equalização do histograma para melhorar a distribuição de cores na imagem
def hsv_histeq(bgr):
    value=90
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    imgeq=cv2.equalizeHist(v)
    final_hsv = cv2.merge((h, s, imgeq))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# Face Mesh Configs
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# FER Emotion
current_emotion=None

# Parâmetros de Face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

id = 0
names = names_list.names


# Câmera
cap = cv2.VideoCapture(0)
cap.set(3, 640) 
cap.set(4, 480) 

minW = 0.1*cap.get(3)
minH = 0.1*cap.get(4)


total_frames = 0

unfocused_frames = 0

times_sleeping=0

happy_frames = 0
sad_frames=0
angry_frames = 0
disgust_frames = 0
fear_frames = 0
surprise_frames=0
neutral_frames=0

def show_json(id, total_frames,unfocused_frames,times_sleeping,happy_frames,sad_frames, 
                angry_frames, disgust_frames, fear_frames, surprise_frames, neutral_frames):

    json_data = {
        id : [
        {
            "date" : datetime.now().strftime("%Y-%m-%d_%H:%M"),
            "focus_index" : 1-(unfocused_frames/total_frames),
            "emotions" : {
                "happy": happy_frames/total_frames,
                "sad" : sad_frames/total_frames,
                "angry" : angry_frames/total_frames,
                "disgust" : disgust_frames/total_frames,
                "fear" : fear_frames/total_frames,
                "surprise" : surprise_frames/total_frames,
                "neutral" : neutral_frames/total_frames
            },
            "times_sleeping":times_sleeping
        }
        ]
    }

    print(json_data)
    return json_data

def save_json(id, total_frames,unfocused_frames,times_sleeping,happy_frames,sad_frames, 
                angry_frames, disgust_frames, fear_frames, surprise_frames, neutral_frames):

    json_data = show_json(**locals())
    filename = 'history.txt'
    
    with open(os.path.join(sys.path[0], filename),'r+') as file:
        print(os.stat(filename).st_size)
        if os.stat(filename).st_size > 0:
            file_data = json.load(file)

            # Se já existem logs do usuário
            if id in file_data:
                file_data[id].append(json_data[id][0])
                file.seek(0)
                json.dump(file_data, file, indent = 4)
            # Se não existem logs do usuário
            else:
                file_data.update(json_data)
                print(file_data)
                file = open(filename,'w+')
                json.dump(file_data, file, indent = 4)
        # Se o arquivo estiver vazio
        else:
            json.dump(json_data, file)
        

    print("History saved successfully!")
            
    
# Tecla H mostra as estaísticas atuais no terminal
# Tecla S salva em um arquivo json
kd.on_press_key("H",lambda _:show_json(id, total_frames,unfocused_frames,times_sleeping,happy_frames,sad_frames, angry_frames, disgust_frames, fear_frames, surprise_frames, neutral_frames))
kd.on_press_key("S",lambda _:save_json(id, total_frames,unfocused_frames,times_sleeping,happy_frames,sad_frames, angry_frames, disgust_frames, fear_frames, surprise_frames, neutral_frames))


while True:
    unfocused_flag=False
    total_frames+=1

    # Flag para checar se pessoa está olhando para fora da tela
    outside_screen_look=False

    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Criando face mesh
    results = faceMesh.process(imgRGB)
    results_face = face_detection.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # Desenha o mesh e landmarks na face da pessoa
            mpDraw.draw_landmarks(img, faceLms,mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

            for id,lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)

              
            # Checando se a pessoa piscou ou dormiu
            d_eye = math.sqrt( (faceLms.landmark[EYES_LIST[0]].x - faceLms.landmark[EYES_LIST[1]].x)**2 + (faceLms.landmark[EYES_LIST[0]].y - faceLms.landmark[EYES_LIST[1]].y)**2)
            d_head = math.sqrt( (faceLms.landmark[HEAD_LIST[0]].x - faceLms.landmark[HEAD_LIST[1]].x) ** 2 + (faceLms.landmark[HEAD_LIST[0]].y - faceLms.landmark[HEAD_LIST[1]].y) ** 2)
            proportional_d = d_eye/d_head
            print(proportional_d)

            if proportional_d < PISCA_THRES and not PISCA_FLAG:
                n_pisc += 1
                print('piscou', n_pisc, 'vezes')
                t = time.time()
                PISCA_FLAG = True

            if PISCA_FLAG:
                dif_t = time.time() - t
                print('Olhos fechados... tempo = ' + str(int(dif_t)) + 's')
                print(dif_t)
                if dif_t >= 3.0:
                    print('ACORDA!!!')
                    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                    unfocused_flag=True
                    times_sleeping+=1

            if proportional_d > PISCA_THRES and PISCA_FLAG:
                PISCA_FLAG = False
                dif_t = time.time() - t


            print("-------------------------")
            print("d_eye", d_eye)
            print(faceLms.landmark[EYES_LIST[0]].x)
            print(faceLms.landmark[EYES_LIST[0]].y)
            print(faceLms.landmark[EYES_LIST[1]])
            print("-------------------------")
            


    # Checando se a pessoa está olhando para o lado
    if results_face.detections:
        for detection in results_face.detections:
            x1 = detection.location_data.relative_bounding_box.xmin # left side of face bounding box
            x2 = x1 + detection.location_data.relative_bounding_box.width # right side of face bounding box

            y1 = detection.location_data.relative_bounding_box.ymin # left side of face bounding box
            y2 = y1 + detection.location_data.relative_bounding_box.height # right side of face bounding box

            cx = (x1 + x2) / 2
            cy = (y1+y2) / 2
            print("Cy: ",cy)
            print(cx)
            if cx < 0.4 or cx > 0.6: # left -> clockwise
                outside_screen_look=True
                print(cx)
                print("Olhou para o lado")
                unfocused_flag=True

            if cy > 0.6: 
                print("looking down")
                unfocused_flag=True

## Live Face Recognition
    # Encontrando Face Limits
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        # Reconhecimento da pessoa com base no treinamento prévio
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(id)

        # Escrevendo nome da pessoa reconhecida
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        cv2.putText(img, str(id), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)  
    
## Emotions detector
    emotions_detector = FER(mtcnn=True)
    img_emotions = hsv_histeq(img)
    emotions_detector.detect_emotions(img_emotions)
    dominant_emotion, emotion_score = emotions_detector.top_emotion(img_emotions)  # seleciona emotion com maior score

    if dominant_emotion is not None:
        if emotion_score is not None:
            current_emotion=dominant_emotion
            current_score=emotion_score
            print(dominant_emotion, emotion_score)

            if current_emotion=="happy":
                happy_frames+=1
            elif current_emotion == "sad":
                sad_frames+=1
            elif current_emotion == "angry":
                angry_frames+=1
            elif current_emotion == "disgust":
                disgust_frames+=1
            elif current_emotion == "fear":
                fear_frames+=1
            elif current_emotion == "surprise":
                surprise_frames+=1
            elif current_emotion == "neutral":
                neutral_frames+=1

        cv2.putText(img, f'{(current_emotion)} : {current_score}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    
    if unfocused_flag:
        unfocused_frames+=1

    cv2.imshow("Test", img)
    #cv2.waitKey(1)

    key=cv2.waitKey(1)
    print("ola")
    if key%256 == 27:
            print("oiee")
            break

   # if cv2.waitKey(1) == ord('q'):
    #    print("Tchau")
    #    break

print("tchau")
cap.release()
cv2.destroyAllWindows()



# ------------ Fontes e tutoriais ------------ #

# Face Mesh
#https://www.youtube.com/watch?v=V9bzew8A1tc

# Emotions detector
# https://pypi.org/project/fer/
# Face recognition:
# https://docs.opencv.org/3.4/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#acc42e5b04595dba71f0777c7179af8c3
# https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826

# Sleep Alarm
# https://github.com/Seungeun-Song/face_recognition
    
# Para rodar: 
    #pip install FER
    #pip install tensorflow
