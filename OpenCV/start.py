import numpy as np
import cv2
# Captures video from the webcam
cap = cv2.VideoCapture(0)  
# This code initiates an infinite loop (to be broken later by a break statement), where we have ret and frame being defined as the cap.read(). Basically, ret is a boolean regarding whether or not there was a return at all, at the frame is each frame that is returned. If there is no frame, you wont get an error, you will get None. 
while(True):
    ret, frame = cap.read()
    # Here, we define a new variable, gray, as the frame, converted to gray. Notice this says BGR2GRAY. It is important to note that OpenCV reads colors as BGR (Blue Green Red), where most computer applications read as RGB (Red Green Blue). Remember this.Here, we define a new variable, gray, as the frame, converted to gray. Notice this says BGR2GRAY. It is important to note that OpenCV reads colors as BGR (Blue Green Red), where most computer applications read as RGB (Red Green Blue). Remember this.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()