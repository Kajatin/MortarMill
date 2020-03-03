import cv2 as cv
import numpy as np
from scipy.stats import kurtosis, norm

row = 0

def row_selection_cb(val):
    global row
    row = int(val)

i = 6
image = cv.imread(f'samples/RAW/brick_{i}.jpg')
image = cv.resize(image, (600, int(image.shape[0] * (600.0/image.shape[1]))))
image_row = image.copy()
image_col = image.copy()

#if src is None:
#    print('Could not open or find the image:', args.input)
#    exit(0)

cv.namedWindow('Source image')
cv.createTrackbar('row_selection','Source image',0,image.shape[0]-1,row_selection_cb)

while(1):
    src = image.copy()
    bgr_planes = cv.split(src)
    gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    #cv.GaussianBlur(gray,(5,5),0,gray)

    cv.imshow('red',bgr_planes[2])
    cv.imshow('green',bgr_planes[1])
    cv.imshow('blue',bgr_planes[0])

    col_width = 100
    bgr_planes[2] = bgr_planes[2][row,:col_width]
    bgr_planes[1] = bgr_planes[1][row,:col_width]
    bgr_planes[0] = bgr_planes[0][row,:col_width]

    cv.line(src,(0,row),(col_width,row),(255,0,0),2)
    #cv.line(bgr_planes[2],(row,0),(row,bgr_planes[2].shape[1]-1),(255,0,0),2)
    #cv.line(bgr_planes[1],(row,0),(row,bgr_planes[1].shape[1]-1),(255,0,0),2)
    #cv.line(bgr_planes[0],(row,0),(row,bgr_planes[0].shape[1]-1),(255,0,0),2)

    histSize = 256
    histRange = (0, 256) # the upper boundary is exclusive
    accumulate = False
    b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
    hist_w = 512
    hist_h = 400
    bin_w = int(np.round( hist_w/histSize ))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    for i in range(1, histSize):
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(np.round(b_hist[i-1])) ),
                ( bin_w*(i), hist_h - int(np.round(b_hist[i])) ),
                ( 255, 0, 0), thickness=2)
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(np.round(g_hist[i-1])) ),
                ( bin_w*(i), hist_h - int(np.round(g_hist[i])) ),
                ( 0, 255, 0), thickness=2)
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(np.round(r_hist[i-1])) ),
                ( bin_w*(i), hist_h - int(np.round(r_hist[i])) ),
                ( 0, 0, 255), thickness=2)
    cv.imshow('Source image', src)
    cv.imshow('calcHist Demo', histImage)
    
    
    gray_hist = cv.calcHist([gray[row,:col_width]], [0], None, [histSize], histRange, accumulate=accumulate)
    histImageGray = np.zeros((hist_h, hist_w, 1), dtype=np.uint8)
    cv.normalize(gray_hist, gray_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    for i in range(1, histSize):
        cv.line(histImageGray, ( bin_w*(i-1), hist_h - int(np.round(gray_hist[i-1])) ),
                ( bin_w*(i), hist_h - int(np.round(gray_hist[i])) ),
                ( 255), thickness=2)

    #kurt_arr = []
    for i in range(gray.shape[0]):
        kurt = kurtosis(gray[i,:],None)
        #print(kurt)
        #kurt_arr.append(kurt)

        if (kurt <= 0):
            cv.line(image_row,(0,i),(col_width,i),(255,0,0),2)

    #kurt_arr = cv.convertScaleAbs(kurt_arr, None, 255.0/np.absolute(kurt_arr).max())
    #cv.imshow('kurt_arr',kurt_arr)

    #for i in range(gray.shape[1]):
    #    kurt = kurtosis(gray[:,i],None)
    #    #print(kurt)

    #    if (kurt > 1):
    #        cv.line(image_col,(i,0),(i,image_col.shape[0]),(255,0,0),2)

    #cv.line(gray,(0,row),(gray.shape[1],row),(255),2)
    cv.line(gray,(0,row),(col_width,row),(255),2)
    cv.imshow('gray hist',histImageGray)
    cv.imshow('gray',gray)
    cv.imshow('im_row',image_row)
    #cv.imshow('im_col',image_col)
    
    
    key = cv.waitKey(30)

    if key == 27:
        exit(0)