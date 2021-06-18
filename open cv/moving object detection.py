#process in object detection
#we should compare the normal frame and the frame with object so that it will identify the target object
#for that we need to do several process
# import necessary libraries
import cv2
import time
import imutils
#lets access camera 
cam = cv2.VideoCapture(0)
#lets initialise sleep time = 1 sec
time.sleep(1)
#Initialise first_frame as none and area = 0 and count = 0
first_frame=None
area = 500
count = 0
while True:
#lets read the image 
    _,img = cam.read()
    text = 'Normal'
#now we resize it using imutils library
    img = imutils.resize(img,width = 500)
#now we convert it to gray image
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#now we convert the gray image into gaussian blurred image
    gaussian_img = cv2.GaussianBlur(gray_img,(21,21),0)
#now we should find the differnce between the two images
    if first_frame is None:
        first_frame = gaussian_img
        continue
    img_diff = cv2.absdiff(first_frame,gaussian_img)
#Then we convert into threshold image
    thresh_img = cv2.threshold(img_diff,25,255,cv2.THRESH_BINARY)[1]
#Then we dilate using dilate function
    thresh_img = cv2.dilate(thresh_img,None,iterations=2)
#Then we should find the contours and grab it for nice full image
    cnts = cv2.findContours(thresh_img.copy(),cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
            count = count+1
            if cv2.contourArea(c) < area:
              continue
#in order to track the object we need to draw the rectangles over it
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            text = 'moving object detected='+str(count)
    print(text)
    cv2.putText(img,text,(10,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow('camerafeed',img)
    key = cv2.waitKey(1)&0xff
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
