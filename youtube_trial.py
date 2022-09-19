

import cv2, pafy

url = 'https://www.youtube.com/watch?v=gdZLi9oWNZg'
video = pafy.new(url)
best  = video.getbest(preftype="mp4")

cap = cv2.VideoCapture(best.url)
frameRate = int(cap.get(cv2.CAP_PROP_FPS))

while True:
    
    ret, frame = cap.read()
    
    cv2.rectangle(frame,(0,0),(120,30),(0,0,0),-1)

    cv2.putText(frame,'Frame: {}'.format(frameRate),(0,15),cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255),1)

    cv2.imshow('frame',frame)

    key = cv2.waitKey(frameRate)
    
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()