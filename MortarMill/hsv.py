from __future__ import print_function
import cv2 as cv
import numpy as np

max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 35
low_V = 180
high_H = 180
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)


#cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

i = 6
image = cv.imread(f'samples/RAW/brick_{i}.jpg')
image = cv.resize(image, (600, int(image.shape[0] * (600.0/image.shape[1]))))
while True:
    frame = image.copy()

    # hsv thresholding
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    # morphology
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    closing = cv.morphologyEx(frame_threshold, cv.MORPH_CLOSE, kernel)
    #kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    #opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel_open)
        
    #cv.imshow(window_capture_name, frame)
    #cv.imshow("close",closing)
    #cv.imshow("open",opening)
    cv.imshow(window_detection_name, frame_threshold)


    # connected components
    nb_comp, label, stats, centroids = cv.connectedComponentsWithStats(closing, connectivity=8)
    min_size = 250

    cc_out = np.zeros((closing.shape))
    for i in range(1,nb_comp):
        if stats[i, -1] >= min_size:
            cc_out[label == i] = 255

                       
    #cv.imshow('connected components', cc_out)

    # connected components inverted
    inverted_image = np.zeros((cc_out.shape),np.uint8)
    inverted_image[cc_out == 0] = 255

    nb_comp_i, label_i, stats_i, centroids_i = cv.connectedComponentsWithStats(inverted_image, connectivity=8)
    min_size = 2000

    cc_out_inverted = np.zeros((closing.shape))
    for i in range(1,nb_comp_i):
        if stats_i[i,-1] >= min_size:
            cc_out_inverted[label_i == i] = 255

    #cv.imshow('connected components inverted', cc_out_inverted)

    # final mask
    final_mask = np.zeros((cc_out_inverted.shape),np.uint8)
    final_mask[cc_out_inverted == 0] = 255
    cv.imshow('final mask', final_mask)
    
    mask = final_mask
    res = cv.bitwise_and(frame,frame,mask=mask)
    
    nb_comp_f, label_f, stats_f, centroids_f = cv.connectedComponentsWithStats(mask, connectivity=8)
    for center in centroids_f[1:]:
        cv.drawMarker(res, tuple(np.uint(center)), (255,0,0), cv.MARKER_CROSS)

    for stat in np.uint(stats_f[1:]):
        cv.rectangle(res, (stat[0],stat[1]),(stat[0]+stat[2],stat[1]+stat[3]),(0,255,0))
    
    cv.imshow("mask",res)


    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break