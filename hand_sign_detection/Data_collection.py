import cv2
import Hand_detection_module as hdm
import numpy as np
import math
import time

wCam, hCam = 1280, 720

source = 2
video_cap = cv2.VideoCapture(source)
video_cap.set(3, wCam)
video_cap.set(4, hCam)

detector = hdm.handDetector(maxHands=1)
offset = 20
imgSize = 500

folder = "data/Right"
counter=0
while True:
    has_frame, frame = video_cap.read()
    if not has_frame:
      break

    hands, frame = detector.boxHands(frame, box=False)

    if hands:
        hand = hands[0]
        x,y,w,h = hand["bbox"]
        imgCropped = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        

        aspectRatio = h/w

        if aspectRatio >1:
            k = imgSize/h
            wCal = math.ceil(k * w)
            imgResized = cv2.resize(imgCropped,(wCal,imgSize))
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResized 
        else:
            k = imgSize/w
            hCal = math.ceil(k * h)
            imgResized = cv2.resize(imgCropped,(imgSize,hCal))
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal+hGap , :] = imgResized 


        cv2.imshow("img cropped", imgCropped)
        cv2.imshow("img white", imgWhite)
    


    # fliped_frame = cv2.flip(frame,1)
    # cv2.imshow("frame", fliped_frame)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
    # Exit the loop.
        break   
    if key == ord('S') or key == ord('s'):
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)

