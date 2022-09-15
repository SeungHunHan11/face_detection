from calendar import c
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os
import uuid
from pathlib import Path
import sys
import argparse

__file__='capture_face.py'
ROOT = Path(os.path.dirname(os.path.realpath('__file__'))).absolute()
sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(os.path.join('/',ROOT))) 

parser = argparse.ArgumentParser()

parser.add_argument(
    '--source',
    type=int,
    default=0,
    help='Select Webcam'
)
parser.add_argument(
    '--proto',
    type=str,
    default=ROOT/'deploy.prototxt',
)
parser.add_argument(
    '--model',
    type=str,
    default=ROOT/'res10_300x300_ssd_iter_140000.caffemodel'
)

args = vars(parser.parse_args())

detector = cv2.dnn.readNetFromCaffe(os.path.join(args['proto']), os.path.join(args['model']))


cap=cv2.VideoCapture(args['source'])

start=time.time()

change_status=True
cnt=0
blur_pic_count=0
unblurred_pic_count=0
while True:
    end=time.time()

    ret,frame = cap.read()
    if not ret:
        break
    

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cv2.rectangle(frame,(0,h-30),(w,h),(0,0,0),-1)
    cv2.rectangle(frame,(0,0),(w,30),(0,0,0),-1)
    
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0, (300,300),(104.0,177.0,123.0), swapRB=False, crop=False)
    
    detector.setInput(imageBlob)
    detections = detector.forward()
    
    for i in range(0,detections.shape[2]):
        
        confidence = detections[0,0,i,2]
        
        if confidence<0.5:
            break
        
        cnt += 1

        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startX, startY, endX, endY) = box.astype("int")
        
        face = frame[startY:endY, startX:endX]
            
        (fH, fW) = face.shape[:2]
            
        if fW < 20 or fH < 20:
            continue
                            
        key = cv2.waitKey(1) & 0xFF
            
        if key==ord('s'):
            change_status=not change_status
                
        if change_status:
            face_image = cv2.GaussianBlur(face,(99,99), 30)
            frame[startY:endY, startX:endX] = face_image
            cv2.rectangle(frame,(startX+2,startY-30),(startX+110,startY-2),(0,200,0),-3)

            cv2.putText(frame,'Blurred',(startX, startY-10),cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255),1)
            cv2.putText(frame, 'Press s to unblur',(w-170,h-10),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 200, 0),1)
            
        else:   
            
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.rectangle(frame,(startX+2,startY-30),(startX+150,startY-2),(0,200,0),-3)
            cv2.putText(frame,'Not Blurred',(startX, startY-10),cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255),1)
            cv2.putText(frame, 'Press s to blur',(w-170,h-10),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 200, 0),1)

    if change_status:
        imgname = os.path.join(ROOT/('vid_rec/blurred/'+str(uuid.uuid1())+'.jpg'))

    else:
        imgname = os.path.join(ROOT/('vid_rec/notblurred/'+str(uuid.uuid1())+'.jpg'))

    if cnt%100==0:
        if change_status:
            
            blur_pic_count+=1
        
        else:
            unblurred_pic_count+=1
        cv2.imwrite(imgname, frame)

    cv2.putText(frame,'# {} Blurred {} Unblurred Image Saved'.format(blur_pic_count,unblurred_pic_count),
                (0,h-10),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),1)

    cv2.putText(frame,'Press Q to quit',(w-140,20),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),1)
    cv2.putText(frame,'Face Detection Module',(10,20),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),1)


    cv2.imshow('Image Collection', frame)
        
    key = cv2.waitKey(1) & 0xFF

    
    if key == ord("q"):
        print('Video Saved, {} seconds recored'.format(round(time.time()-start)))
        break        

cap.release()
cv2.destroyAllWindows()