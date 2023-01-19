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



  def boxHands(self, frame, flipType = True, draw = True,box = True):
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(frame_RGB)
    allHands = []
    h, w, c = frame.shape
    if self.results.multi_hand_landmarks:
      for handType, handlms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
        myHand = {}
        lmList = []
        xList = []
        yList = []
        for id, lm in enumerate(handlms.landmark):
          px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
          lmList.append([px, py, pz])
          xList.append(px)
          yList.append(py)

        #box
        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        boxW, boxH = xmax-xmin, ymax - ymin
        bbox = xmin, ymin, boxW, boxH
        cx, cy = bbox[0] + (bbox[2] // 2), \
                bbox[1] + (bbox[3] // 2)

        myHand["lmList"] = lmList
        myHand["bbox"] = bbox
        myHand["center"] = (cx,cy)

        if flipType:
          if handType.classification[0].label == "Right":
            myHand["type"] = "Left"
          else:
            myHand["type"] = "Right"
        else:
          myHand["type"] = handType.classification[0].label
        allHands.append(myHand)

        #drawing box
        if draw:
          self.mpDraw.draw_landmarks(frame, handlms, self.mpHands.HAND_CONNECTIONS)
          if box:
            cv2.rectangle(frame, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 0, 255), 3)
            cv2.putText(frame, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    if draw:       
      return allHands, frame
    else:
      return allHands



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
