import logging

import cv2 as cv
import numpy as np
import math
from scipy.stats import kurtosis
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from common.FCM import FCM
from vision import imgproc
from trainer import classifier


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PathFinder(metaclass=Singleton):
    """description of class"""

    def __init__(self):
        # lower and upper HSV threshold ranges
        self.lowerb = None
        self.upperb = None
        # Bayes classifier
        self.bayes_clf = classifier.loadClassifier('bayes_clf.pickle')

        self.logger = logging.getLogger(__name__)


        #TODO: implement config parser and read config and evaluate them
        # code here

        #self.calibrateHsvThresholds_()


    def __call__(self, frames):
        self.__processFrames(frames)
        
        
    def calibrateHsvThresholds_(self, frame, supervised=True):
        """
        Finds the best HSV threshold values used to separate the bricks from
        the mortar.

        Parameters
        ----------        
        frame: array
            The array containing the colour image data. The input is interpreted
            as a BGR 3-channel image. If `frame` hasn't got 3 channels, this
            function fails and returns False.

        supervised(Optional): bool
            If set to True, the HSV calibration is performed manually by the user.
            During the process, the user selects the area of the brick. Once the
            calibration is done, the user can press `s` to accept the results,
            `esc` to cancel the calibration, and any other button to try again.
            If False, the calibration is done automatically. The algorithm finds
            the suitable HSV threshold values that separate the bricks from the mortar.

        Returns
        -------
        ret: bool
            Returns True if the calibration is successful, otherwise False.
        """

        # validate the arguments
        if frame is None:
            self.logger.warning(('The HSV threshold calibration is '
                                 'unsuccessful because the input '
                                 'frame is None.'))
            return False

        if len(frame.shape) != 3 or frame.shape[2] != 3:
            self.logger.warning(('The input `frame` does not have the correct'
                                 ' number of channels. The HSV threshold '
                                 'calibration is unsuccessful'))
            return False

        while 1:
            if supervised:
                self.logger.info('Starting HSV threshold calibration in manual mode.')

                # manually select ROI (area of a brick)
                r = cv.selectROI(frame)

                # crop image -> extract the brick pixels
                frame_cropped = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
                # convert to HSV
                frame_hsv = cv.cvtColor(frame_cropped, cv.COLOR_BGR2HSV)
            
                # evaluate histograms
                maxs = []
                colours = ('b','g','r')
                for i,c in enumerate(colours):
                    # calculate histogram for given channel
                    hist = cv.calcHist([frame_hsv], [i], None, [256], [0,256])
                    # append the index of the maximum to array
                    maxs.append(hist.argmax())
                
                    # plot the histogram for given channel
                    plt.plot(hist,color=c)
                    plt.plot(maxs[-1], hist[maxs[-1]], color=c, marker='x')
                    plt.xlim([0,256])

                # show the histogram plot
                plt.show()

            else:
                self.logger.info('Starting HSV threshold calibration in automatic mode.')
                # automatically find brick cluster center

                # convert to HSV
                frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

                # normalize values to 0-1 range
                frame_hsv = frame_hsv.astype(np.float64)
                frame_hsv[:,:,0] /= 179.0
                frame_hsv[:,:,1:3] /= 255.0

                # reshape image into 2D array
                frame_to_fit = frame_hsv.reshape(-1, 3)

                # fit kmeans with 2 classes on the array
                kmeans = KMeans(n_clusters=2).fit(frame_to_fit)

                #TODO: refine this code to find the correct cluster center
                # find the cluster center corresponding to the bricks
                mask = kmeans.labels_.reshape(frame_hsv.shape[0], frame_hsv.shape[1])
                mask = mask.astype(np.uint8)
                
                # refine mask (remove small objects)
                mask = self.connectedComponentsBasedFilter_(mask)

                # determine the number of BLOBs in the final mask
                # if it is a small number the 0-index cluster center is selected
                # otherwise the 1-index cluster center
                nb_comp, *_ = cv.connectedComponentsWithStats(mask, connectivity=8)
                i = 0 if nb_comp < 3 else 1

                self.logger.debug(('Found cluster centers for HSV thresholds: {}'
                                   '. The selected cluster index is {}.')
                                  .format(kmeans.cluster_centers_, i))
                
                # set the `maxs` array with the HSV threshold values
                maxs = kmeans.cluster_centers_[i].copy()

                # scale back up to the correct HSV range
                maxs[0] *= 179
                maxs[1:3] *= 255
                maxs = maxs.astype(np.uint8)

                self.logger.info('Final HSV threshold values: {}'.format(maxs))

            # generate lower and upper HSV bounds
            eps = 30
            lowerb = tuple([int(val-eps) for val in maxs])
            upperb = tuple([int(val+eps) for val in maxs])
            self.logger.debug(('Candidate HSV threshold ranges found (not yet '
                              'accepted). Lower: {} Upper: {}').format(lowerb,upperb))

            if supervised:
                # test the HSV thresholding
                frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                frame_threshold = cv.inRange(frame_HSV, lowerb, upperb)
                cv.imshow('threshold',frame_threshold)

                key = cv.waitKey(0)
                if key == 27:
                    self.logger.warning(('Calibration of the HSV threshold '
                                         'ranges is unsuccessful. The user'
                                         ' did not accept the any of the results.'))
                    break
                elif key != ord('s'):
                    continue

            # save the HSV threshold calibration results
            self.lowerb = lowerb
            self.upperb = upperb
            self.logger.info(('Calibrated the HSV threshold ranges. '
                                'Lower: {}\tupper: {}').format(self.lowerb, self.upperb))
            break

        cv.destroyAllWindows()
        return True


    def connectedComponentsBasedFilter_(self, frame):
        if len(frame.shape) != 2:
            self.logger.error(('This function can only be called with a single '
                               'channel 8-bit image.'))
            return None

        # connected components
        nb_comp, label, stats, centroids = cv.connectedComponentsWithStats(frame, connectivity=8)
        min_size = 250

        cc_out = np.zeros((frame.shape))
        for i in range(1,nb_comp):
            if stats[i, -1] >= min_size:
                cc_out[label == i] = 255

        # connected components inverted
        inverted_image = np.zeros((cc_out.shape),np.uint8)
        inverted_image[cc_out == 0] = 255

        nb_comp_i, label_i, stats_i, centroids_i = cv.connectedComponentsWithStats(inverted_image, connectivity=8)
        min_size = 2000

        cc_out_inverted = np.zeros((frame.shape))
        for i in range(1,nb_comp_i):
            if stats_i[i,-1] >= min_size:
                cc_out_inverted[label_i == i] = 255

        # final mask
        final_mask = np.zeros((cc_out_inverted.shape),np.uint8)
        final_mask[cc_out_inverted == 0] = 255
        #cv.imshow('final mask', final_mask)

        return final_mask


    def locateBricksBayes(self, frame):
        ## copy of original input
        #frame_original = frame.copy()

        ##frame_original = cv.blur(frame_original, (5,5))
        ## Select ROI
        #r = cv.selectROI(frame_original)
        ## Crop image
        #imCrop = frame_original[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        ## convert to HSV
        #brick = cv.cvtColor(imCrop,cv.COLOR_BGR2HSV)
        #brick = brick.reshape(-1,3)
        #brick_y = np.ones(brick.shape[0])

        ## Select ROI
        #r = cv.selectROI(frame_original)
        ## Crop image
        #imCrop = frame_original[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        ## convert to HSV
        #mortar = cv.cvtColor(imCrop,cv.COLOR_BGR2HSV)
        #mortar = mortar.reshape(-1,3)
        #mortar_y = np.zeros(mortar.shape[0])

        #X = np.vstack([brick,mortar])
        #y = np.hstack([brick_y,mortar_y])

        #pred_img = cv.cvtColor(frame_original, cv.COLOR_BGR2HSV)
        #pred_img = pred_img.reshape(-1,3)

        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        pred = self.bayes_clf.predict(frame_hsv.reshape(-1,3))
        pred *= 255
        pred = pred.reshape(frame.shape[:2])
        pred = pred.astype(np.uint8)

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        closing = cv.morphologyEx(pred, cv.MORPH_CLOSE, kernel)

        # generate final mask based on connected component analysis
        mask = self.connectedComponentsBasedFilter_(pred)
        
        cv.imshow('prediction',pred)
        cv.imshow('mask',mask)

        return mask


    def locateBricksHsv(self, frame):
        """
        Segments the input frame based on HSV threshold values. This function
        requires that the lower and upper HSV thresholds are already available.
        Use the `calibrateHsvThresholds_()` function to calibrate them.

        Parameters
        ----------        
        frame: array
            The array containing the colour image data. The input is interpreted
            as a BGR 3-channel image. If `frame` hasn't got 3 channels, this
            function fails and returns False.

        Returns
        -------
        mask: array
            The final mask image array (2 channels, binary). 255 represents brick
            pixels, 0 for the mortar.
        """

        # check if the lower and upper bounds have been set
        if self.lowerb is None or self.upperb is None:
            self.logger.error(('The lower and/or upper HSV threshold bounds'
                               ' were not set. Cannot run HSV segmentation.'))
            return None
        # check that the input frame has 3 channels (BGR)
        if len(frame.shape) != 3 or frame.shape[2] != 3: 
            self.logger.error('The frame input does not contain 3 channels.')
            return None
        
        # hsv thresholding
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_hsv, self.lowerb, self.upperb)

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        closing = cv.morphologyEx(frame_threshold, cv.MORPH_CLOSE, kernel)
        
        # generate final mask based on connected component analysis
        mask = self.connectedComponentsBasedFilter_(closing)

        return mask


    def locateBricksHistogram(self, frame, limit):
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
        frame = image_arg.copy()
        
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, self.lowerb, self.upperb)

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


    def __processFrames(self, frames):
        frame_colour = frames['colour'].copy()
        frame_depth = frames['depth'].copy()

        mask_hsv = self.locateBricksHsv(frame_colour)
        mask_bayes = self.locateBricksBayes(frame_colour)
        #mask_hist = self.locateBricksHistogram(frame_colour, 0)
        #self.locate_bricks(frame_colour,100,160)

        #TODO: implement a better way to combine the masks maybe based on propabilities
        # final mask
        mask = cv.bitwise_or(mask_hsv,mask_bayes)

        # detect bricks with final mask
        frame_colour_copy = frame_colour.copy()
        nb_comp_f, label_f, stats_f, centroids_f = cv.connectedComponentsWithStats(mask, connectivity=8)
        for center in centroids_f[1:]:
            cv.drawMarker(frame_colour_copy, tuple(np.uint(center)), (255,0,0),
                          cv.MARKER_CROSS, thickness=2)

        for stat in np.uint(stats_f[1:]):
            cv.rectangle(frame_colour_copy,
                         (stat[0],stat[1]),(stat[0]+stat[2],stat[1]+stat[3]),
                         (0,0,255), 2)

        # detect contours
        cnts, hrcy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(frame_colour_copy, cnts, -1, (0,255,0),2)
    
        # show brick detection results
        masked_orig = cv.bitwise_and(frame_colour,frame_colour,mask=np.invert(mask))
        cv.imshow('output',np.vstack((frame_colour_copy,masked_orig,frame_colour)))




        #########################################################

        #mask_hsv = self.locateBricksHsv(frame_colour)
        #mask_hist = self.locate_bricks_hist(frame_colour, 0)
        ##mask_plane, mask_ransac = imgproc.separateDepthPlanes(frame_depth)

        #final_mask = cv.bitwise_not(mask_hsv, mask_hist)
        
        #processed_frame = cv.bitwise_and(frame_colour, frame_colour, mask=final_mask)
        #cv.imshow('Final mask', final_mask)
        #cv.imshow('Final masked image', processed_frame)