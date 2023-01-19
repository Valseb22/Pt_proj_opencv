import cv2
import time
import mediapipe as mp

wCam, hCam = 1180, 720
source=2
video_cap = cv2.VideoCapture(source)
video_cap.set(3, wCam)
video_cap.set(4, hCam)
pTime=0
cTime=0
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw=mp.solutions.drawing_utils


while True:
    has_frame, frame = video_cap.read()
    if not has_frame:
      break

    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_RGB)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print (id, cx, cy)
                cv2.circle(frame,(cx, cy), 10, (255, 255, 0), cv2.FILLED)
            mpDraw.draw_landmarks(frame, handlms, mpHands.HAND_CONNECTIONS)

    flipped_frame=cv2.flip(frame,1)
    cv2.imshow("Test", frame)
    cv2.imshow("flipped", flipped_frame)
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
    # Exit the loop.
        break   


