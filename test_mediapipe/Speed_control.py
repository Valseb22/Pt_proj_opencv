import cv2
import numpy as np
import Hand_detection_module as hdm
import math

# wCam, hCam = 1920, 1080
wCam, hCam = 1080, 720



source = 2
video_cap = cv2.VideoCapture(source)
video_cap.set(3, wCam)
video_cap.set(4, hCam)

detector = hdm.handDetector()

minSpeed = 0
maxSpeed = 100
speedBar = 0




while True:
    has_frame, frame = video_cap.read()
    if not has_frame:
      break

    hands, frame = detector.boxHands(frame, box = False)
    #lmList= detector.findPosition(frame,handNbr=1 ,draw = False) a débug, main droite: handNbr=1
    lmList= detector.findPosition(frame,handNbr=0 ,draw = False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2)//2 , (y1 + y2)//2

        cv2.circle(frame, (x1, y1), 10, (255, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 10, (255, 255, 0), cv2.FILLED)
        
        cv2.line(frame, (x1,y1), (x2,y2), (255,255, 0), 3)

        length = math.hypot(x2 - x1, y2 - y1)
        #print(length)



        if length<30:
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        elif length>300:
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        else:
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        speed= np.interp(length, [30,300], [minSpeed, maxSpeed])
        speedBar= np.interp(length, [30,300], [335, 75])
        #print(speed)

    cv2.rectangle(frame, (75, 75), (125, 335), (0, 255, 0), 3)
    cv2.rectangle(frame, (75, int(speedBar) ), (125, 335), (0, 255, 255), cv2.FILLED)



    fliped_frame = cv2.flip(frame,1)
    cv2.imshow("frame", fliped_frame)
    #cv2.imshow("speed Control", frame)
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
    # Exit the loop.
        break   
