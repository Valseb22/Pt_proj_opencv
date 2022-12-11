import cv2
import time


source=2
video_cap = cv2.VideoCapture(source)
pTime=0

while True:
    has_frame, frame = video_cap.read()
    if not has_frame:
      break

    cTime=time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS:{int(fps)}',(20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Test", frame)
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
    # Exit the loop.
        break
