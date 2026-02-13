import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to images folder
path = 'images'
images = []
classNames = []

myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img,model="hog")[0]# trying hog       
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

#  attendance
def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# cam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)#LOWER RESOLUTION
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,15)

frame_count = 0
SKIP_FRAMES = 5

frame_count = 0
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS,model="hog")
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame,model="hog")

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, name, (x1,y2+35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            markAttendance(name)

    cv2.imshow('Face Attendance System', img)
    if cv2.waitKey(1)&0xFF == 13:  #  Enter to exit
       break

cap.release()
cv2.destroyAllWindows()
