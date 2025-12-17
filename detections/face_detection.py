import cv2

cap=cv2.VideoCapture(0)
frontalface_detection=cv2.CascadeClassifier("../haarcascade/haarcascade_frontalface_default.xml")


while True:
    ret, frame=cap.read()
    video=cv2.flip(frame,1)
    gray_video=cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)


    #FACE DETECTION
    face=frontalface_detection.detectMultiScale(gray_video,1.2,6)
    for (x,y,w,h) in face:
        cv2.rectangle(video,(x,y),(x+w,y+h),(0,0,255),5)
        print(int(x+w),int(y+h))

    cv2.imshow('face_detection', video)

    key = cv2.waitKey(15) & 0xff
    if key == ord('q'):
        break
                              
cv2.destroyAllWindows()



