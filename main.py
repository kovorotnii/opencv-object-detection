""" Pedestrian detection """
import cv2
# import numpy as np

cap = cv2.VideoCapture('static-cctv.mp4')
#cap = cv2.VideoCapture('dust-pedestrians.mp4')
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame1, "FPS: {}".format(fps), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 0), 2)
        # calculate diff on two frames
        diff = cv2.absdiff(frame1, frame2)
        # convert to shades of gray
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
        _,thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 1500:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (255, 144, 30),2)

            cv2.putText(frame1, "State: {}".format('Movement'), 
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0),2)
        # draw all movement objects contours
        #cv2.drawContours(frame1, contours, -1, (0,0,0), 2)
    
        cv2.imshow('Pedestrians', frame1)
        frame1 = frame2

        ret, frame2 = cap.read()

        if cv2.waitKey(40) == 27:
            break
    except cv2.error as error:
        print("Open CV error: {}", error)
cv2.destroyAllWindows()
cap.release()