import numpy as np
import cv2

facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('c1.mp4')

#Mr.Selfridge.S04E10.HDTV.x264-ORGANiC[eztv].mp4
while(cap.isOpened()):
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        sub_face=img[y:y+h,x:x+w]
        sub_face = cv2.GaussianBlur(sub_face, (23, 23), 80)
       # blur=cv2.GaussianBlur(img,(15,15),30)


        cv2.imshow("Face", img)
        cv2.imshow("blur",sub_face)



    if (cv2.waitKey(30) & 0xFF == ord('q')):
        break;

cap.release()
cv2.destroyAllWindows()