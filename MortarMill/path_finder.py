import cv2 as cv
import numpy as np
import math
from scipy.stats import kurtosis

from sklearn.cluster import KMeans

from common.FCM import FCM
from vision import imgproc


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PathFinder(metaclass=Singleton):
    """description of class"""

    def __init__(self):
        pass


    def locate_bricks_hsv(self, frame):
        low_H = 0
        low_S = 35
        low_V = 180
        high_H = 180
        high_S = 255
        high_V = 255

        # copy of original input
        frame_original = frame.copy()
        
        # hsv thresholding
        frame_HSV = cv.cvtColor(frame_original, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        closing = cv.morphologyEx(frame_threshold, cv.MORPH_CLOSE, kernel)
        
        # connected components
        nb_comp, label, stats, centroids = cv.connectedComponentsWithStats(closing, connectivity=8)
        min_size = 250

        cc_out = np.zeros((closing.shape))
        for i in range(1,nb_comp):
            if stats[i, -1] >= min_size:
                cc_out[label == i] = 255

        # connected components inverted
        inverted_image = np.zeros((cc_out.shape),np.uint8)
        inverted_image[cc_out == 0] = 255

        nb_comp_i, label_i, stats_i, centroids_i = cv.connectedComponentsWithStats(inverted_image, connectivity=8)
        min_size = 2000

        cc_out_inverted = np.zeros((closing.shape))
        for i in range(1,nb_comp_i):
            if stats_i[i,-1] >= min_size:
                cc_out_inverted[label_i == i] = 255

        # final mask
        final_mask = np.zeros((cc_out_inverted.shape),np.uint8)
        final_mask[cc_out_inverted == 0] = 255
        #cv.imshow('final mask', final_mask)
    
        mask = final_mask
        res = cv.bitwise_and(frame_original,frame_original,mask=mask)
    
        # detect bricks
        nb_comp_f, label_f, stats_f, centroids_f = cv.connectedComponentsWithStats(mask, connectivity=8)
        for center in centroids_f[1:]:
            cv.drawMarker(frame_original, tuple(np.uint(center)), (255,0,0), cv.MARKER_CROSS,thickness=2)

        for stat in np.uint(stats_f[1:]):
            cv.rectangle(frame_original, (stat[0],stat[1]),(stat[0]+stat[2],stat[1]+stat[3]),(0,0,255),2)

        # detect contours
        cnts, hrcy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(frame_original, cnts, -1, (0,255,0),2)
    
        #cv.imshow('mask',mask)
        #cv.imshow('output',np.vstack((frame_original,res,frame)))
        #cv.imwrite('output.png',np.hstack((res,frame_original)))

        return mask


    def locate_bricks_hist(self, frame, limit):
        image_row = frame.copy()
        image_col = frame.copy()

        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

        # horizontal
        rows = []
        for i in range(gray.shape[0]):
            kurt = kurtosis(gray[i,],None)
            
            if kurt <= limit:
                rows.append(i)
                cv.line(image_col,(0,i),(gray.shape[1],i),(0,255,0),1)

        edges = []
        threshold = 10
        prev = -1
        for i,row in enumerate(rows):
            if row - prev > threshold:
                edges.append(rows[i-1])
                edges.append(row)
            prev = row

        for edge in edges:
            cv.line(image_row,(0,edge),(gray.shape[1],edge),(0,0,255),2)

        mask = np.ones(frame.shape[:2], np.uint8) * 255
        mask[rows,] = 0

        #cv.imshow('Histogram based segmentation',np.vstack([frame, image_row, image_col]))
        #cv.imshow('Mask Histogram',mask)

        return mask


    def locate_bricks(self, image_arg, min, max):
        low_H = 0
        low_S = 20
        low_V = 180
        high_H = 20
        high_S = 255
        high_V = 255

        frame = image_arg.copy()
        
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
        closing = cv.morphologyEx(frame_threshold, cv.MORPH_CLOSE, kernel)
        opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

        mask = closing
        res = cv.bitwise_and(frame,frame,mask=mask)
        #cv.imshow("mask",mask)
        
        
        #blurred = cv.GaussianBlur(image_gray, (9,9), 0)
        #cv.imshow('blurred',np.hstack((frame, blurred)))
        #cv.waitKey(0)
        
        im_selected = mask

        canny = cv.Canny(im_selected, min, max)
        cv.imshow('canny',np.hstack((im_selected, canny)))

        #lines = cv.HoughLinesP(frame, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]])
        lines = cv.HoughLinesP(canny, 1, math.pi/2, 2, None, 10, 3);

        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    pt1 = (x1,y1)
                    pt2 = (x2,y2)
                    cv.line(frame, pt1, pt2, (0,0,255), 3)

        cv.imshow('hough',frame)


        laplace = cv.Laplacian(im_selected, cv.CV_64F, ksize=3)
        laplace = cv.convertScaleAbs(laplace, None, 255.0/np.absolute(laplace).max())
        cv.imshow('laplace',laplace)


        sobelx = cv.Sobel(im_selected,cv.CV_64F,1,0,None,5)
        sobely = cv.Sobel(im_selected,cv.CV_64F,0,1,None,5)

        sobelx = cv.convertScaleAbs(sobelx, None, 255.0/np.absolute(sobelx).max())
        sobely = cv.convertScaleAbs(sobely, None, 255.0/np.absolute(sobely).max())

        cv.imshow('sobelx',sobelx)
        cv.imshow('sobely',sobely)

        sobel_added = sobelx + sobely
        sobel_added = cv.convertScaleAbs(sobel_added, None, 255.0/np.absolute(sobel_added).max())
        cv.imshow('sobel_added',sobel_added)

        threshold = 225

        #sobelx_bin = cv.adaptiveThreshold(sobelx,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        ret, sobelx_bin = cv.threshold(sobelx, threshold, 255, cv.THRESH_BINARY)
        #ret, sobelx_bin = cv.threshold(sobelx, threshold, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imshow('sobelx_bin', sobelx_bin)

        #sobely_bin = cv.adaptiveThreshold(sobely,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        ret, sobely_bin = cv.threshold(sobely, threshold, 255, cv.THRESH_BINARY)
        #ret, sobely_bin = cv.threshold(sobely, threshold, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imshow('sobely_bin', sobely_bin)

        #sobel_added_bin = cv.adaptiveThreshold(sobel_added,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        ret, sobel_added_bin = cv.threshold(sobel_added, threshold, 255, cv.THRESH_BINARY)
        #ret, sobel_added_bin = cv.threshold(sobel_added, threshold, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imshow('sobel_added_bin', sobel_added_bin)


        # morphology
        #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        #sobely_erosion = cv.morphologyEx(sobely_bin, cv.MORPH_OPEN, kernel)
        #cv.imshow('sobely_erosion', sobely_erosion)

        return None


    def processFrames(self, frames):
        frame_colour = frames['colour'].copy()
        frame_depth = frames['depth'].copy()

        mask_hsv = self.locate_bricks_hsv(frame_colour)
        mask_hist = self.locate_bricks_hist(frame_colour, 0)
        #mask_plane, mask_ransac = imgproc.separateDepthPlanes(frame_depth)

        final_mask = cv.bitwise_not(mask_hsv, mask_hist)
        
        processed_frame = cv.bitwise_and(frame_colour, frame_colour, mask=final_mask)
        cv.imshow('Final mask', final_mask)
        cv.imshow('Final masked image', processed_frame)