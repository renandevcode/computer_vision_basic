import cv2

image=cv2.imread("../images/carsOnTheRoad.jpg")
image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
car_detection=cv2.CascadeClassifier("../haarcascade/cars.xml")

car=car_detection.detectMultiScale(image_gray,1.1,3,minSize=(60,60))
for(x,y,w,h) in car:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),4)


cv2.imshow('cars',image)
cv2.waitKey(0)

cv2.destroyAllWindows()
