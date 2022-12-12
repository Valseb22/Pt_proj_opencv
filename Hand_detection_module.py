import cv2
import time
import mediapipe as mp

class handDetector():
  def __init__(self, mode=False, maxHands=2, modelComp= 1, detectionCon=0.5, trackCon=0.5):
    self.mode = mode
    self.maxHands = maxHands
    self.modelComp = modelComp
    self.detectionCon = detectionCon
    self.trackCon = trackCon
    self.mpHands = mp.solutions.hands
    self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelComp,self.detectionCon,self.trackCon)
    self.mpDraw = mp.solutions.drawing_utils


  def findHands(self, frame, draw = True):
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(frame_RGB)
    if self.results.multi_hand_landmarks:
        for handlms in self.results.multi_hand_landmarks:
          if draw:
            self.mpDraw.draw_landmarks(frame, handlms, self.mpHands.HAND_CONNECTIONS)
    return frame


  def findPosition(self, frame, handNbr=0, draw= True):
    lmList = []
    if self.results.multi_hand_landmarks:
      myHand = self.results.multi_hand_landmarks[handNbr]
      for id, lm in enumerate(myHand.landmark):
          h, w, c = frame.shape
          cx, cy = int(lm.x*w), int(lm.y*h)
          lmList.append([id, cx, cy])
          if draw:
            cv2.circle(frame,(cx, cy), 10, (255, 255, 0), cv2.FILLED)

    return lmList






def main():
  source = 2
  video_cap = cv2.VideoCapture(source)
  detector = handDetector()

  while True:
    has_frame, frame = video_cap.read()
    if not has_frame:
      break

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame)
    if len(lmList) != 0:
      print(lmList[12])
    flipped_frame=cv2.flip(frame,1)

    cv2.imshow("flipped", flipped_frame)
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
    # Exit the loop.
        break   




if __name__ == "__main__":
  main()
