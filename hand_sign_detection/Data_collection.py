import cv2
import Hand_detection_module as hdm
import numpy as np
import math
import time

wCam, hCam = 1180, 720

source = 2
video_cap = cv2.VideoCapture(source)
video_cap.set(3, wCam)
video_cap.set(4, hCam)

detector = hdm.handDetector(maxHands=1)
offset = 20
imgSize = 500

folder = "data/images"
folder_lbl = "data/label"
counterL=0
counterR=0
counterO=0
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

        [h,w,dim]=imgResized.shape
        answ=input('type image? \n r:right \n l:left \n o:other %s\n')
        if answ=='r':
            counterR+=1
            cv2.imwrite(f'{folder}/Image_right_{counterR}.jpg',imgWhite)
            with open(f'{folder_lbl}/Image_right_{counterR}.txt', "w") as file:
                file.write(f'0 0.5 0.5 {w/500} {h/500}')
        elif answ=='l':
            counterL+=1
            cv2.imwrite(f'{folder}/Image_left_{counterL}.jpg',imgWhite)
            with open(f'{folder_lbl}/Image_left_{counterL}.txt', "w") as file:
                file.write(f'1 0.5 0.5 {w/500} {h/500}')
        else:
            counterO+=1
            cv2.imwrite(f'{folder}/Image_other_{counterO}.jpg',imgWhite)
            with open(f'{folder_lbl}/Image_other_{counterO}.txt', "w") as file:
                file.write(f'2 0.5 0.5 {w/500} {h/500}')
        print(f'nb images right: {counterR} \n nb images left:{counterL} \n nb images other: {counterO} \n')



