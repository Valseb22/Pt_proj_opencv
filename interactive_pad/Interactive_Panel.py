import cv2
import numpy as np

drawing_points=[0,0]


def draw_contours(image, hsv_max, hsv_min ):
    global drawing_points
    ksize=(5, 5)

    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(frame_hsv, HSV_min, HSV_max)
    mask_erode = cv2.erode(mask, np.ones(ksize, np.uint8))
    contours, hierarchy= cv2.findContours(mask_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        ((x,y),radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(image, (int(x),int(y)),int(round(radius)),(0,255,0),3)
        #if (len(cnt)>0):
        #M=cv2.moments(cnt)
        #mx=int(M["m10"]) #/ M["m00"])
        #my=int(M["m01"]) # / M["m00"])
        drawing_points.append(int(x))
        drawing_points.append(int(y))
    #cv2.drawContours(image, contours, -1, (0,0,255),3)

    return image



def draw_image(image, drawing_points):
    colored_image=image
    if (len(drawing_points)>1001):
        del drawing_points[0:98]
    for i in range(len(drawing_points)-1):
        if ((i % 2)==0):
            if drawing_points[i]!=0:
                colored_image = cv2.circle(colored_image,(drawing_points[i], drawing_points[i+1]),10,(0,255,0),-1)
                #=cv2.circle(image,(drawing_points[i], drawing_points[i+1]),10,(0,255,0),-1)
        #colored_image = cv2.circle(image,(10, 10),10,(0,255,0),-1)
    return colored_image




source = 2
video_cap = cv2.VideoCapture(source)



HSV_max=(32, 192, 226)
HSV_min=(13, 135, 163)
while 1:
    has_frame, frame = video_cap.read()
    if not has_frame:
      break
    frame = cv2.flip(frame,1)

    erode = draw_contours(frame,HSV_max, HSV_min)

    cv2.imshow('pen detection', erode)
    #cv2.imshow('HSV',frame_hsv)
    #cv2.imshow('simple_mask',mask)
    panel=draw_image(frame,drawing_points)
    cv2.imshow('interactive panel', panel)
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
    # Exit the loop.
        break