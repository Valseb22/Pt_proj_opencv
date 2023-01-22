import cv2
import numpy as np


max_value = 255
max_val_H = 180
low_H = 0
low_S = 0
low_V = 0
high_V = max_value
high_S = max_value
high_H = max_val_H
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
window_detection_name = 'Object Detection'


def on_H_min_trackbar(val):
  global low_H
  global high_H
  low_H=val
  low_H=min(high_H-1,low_H)
  cv2.setTrackbarPos(low_H_name,window_detection_name, low_H)

def on_H_max_trackbar(val):
  global low_H
  global high_H
  high_H=val
  high_H=max(high_H,low_H+1)
  cv2.setTrackbarPos(high_H_name,window_detection_name, high_H)

def on_S_min_trackbar(val):
  global low_S
  global high_S
  low_S=val
  low_S=min(high_S-1,low_S)
  cv2.setTrackbarPos(low_S_name,window_detection_name, low_S)

def on_S_max_trackbar(val):
  global low_S
  global high_S
  high_S=val
  high_S=max(high_S,low_S+1)
  cv2.setTrackbarPos(high_S_name,window_detection_name, high_S)
  
def on_V_min_trackbar(val):
  global low_V
  global high_V
  low_V=val
  low_V=min(high_V-1,low_V)
  cv2.setTrackbarPos(low_V_name,window_detection_name, low_V)

def on_V_max_trackbar(val):
  global low_V
  global high_V
  high_V=val
  high_V=max(high_V,low_V+1)
  cv2.setTrackbarPos(high_V_name,window_detection_name, high_V)


source = 2
video_cap = cv2.VideoCapture(source)


cv2.namedWindow(window_detection_name)

cv2.createTrackbar(low_H_name, window_detection_name, low_H, max_val_H, on_H_min_trackbar)
cv2.createTrackbar(high_H_name, window_detection_name, high_H, max_val_H, on_H_max_trackbar)
cv2.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_S_min_trackbar)
cv2.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_S_max_trackbar)
cv2.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_V_min_trackbar)
cv2.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_V_max_trackbar)
while True:
    has_frame, frame = video_cap.read()
    if not has_frame:
      break

    #frame_HSV=cv2.cvtColor(imboat,cv2.COLOR_BGR2HSV)
    frame_HSV=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    #mask=cv2.inRange(frame,RGB_max,RGB_min)
    #
    #cv2.imshow(win_name, imboat)
    #cv2.imshow(window_detection_name,)
    cv2.imshow('HSV Video', frame_HSV)
    cv2.imshow('mask',frame_threshold)
    #cv2.imshow('mask',mask_hsv)
    #frame_composite= np.hstack([frame_HSV, frame_threshold])
    #cv2.imshow('blabla',frame_composite)


    key = cv2.waitKey(30)
    if key == ord('Q') or key == ord('q') or key == 27:
    # Exit the loop.
        break


