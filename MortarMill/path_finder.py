# builtins
import logging

# 3rd party
import cv2 as cv
import numpy as np
from scipy.stats import kurtosis

# custom
import vision
import trainer


class Singleton(type):
    """ Metaclass used to create a singleton class instance """

    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PathFinder(metaclass=Singleton):
    """description of class"""

    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.logger.debug('Creating PathFinder instance.')

        # debug mode
        self.debug = False
        # lower and upper HSV threshold ranges
        self.lowerb = None
        self.upperb = None
        # bayes classifier
        self.bayes_clf = None
        # svm classifier
        self.svm_clf = None

        # parse the config file if it is not None
        if config is None:
            self.logger.warning(('No config is provided to PathFinder. Some '
                                 'parameters might not be available, and this '
                                 'module might not work correctly.'))
        else:
            self.logger.debug('Parsing config for PathFinder. {}'.format(config))

            self.debug = config.getboolean('debug', False)

            lowerb = (config.getint('min_h'),config.getint('min_s'),config.getint('min_v'))
            upperb = (config.getint('max_h'),config.getint('max_s'),config.getint('max_v'))
            self.lowerb = lowerb if all(lowerb) else None
            self.upperb = upperb if all(upperb) else None

            self.bayes_clf = trainer.classifier.loadClassifier(config.get('bayes'))

            self.svm_clf = trainer.classifier.loadClassifier(config.get('svm'))


    def __call__(self, frames):
        self.__processFrames(frames)


    def locateBricksBayes(self, frame, hsv=False):
        """
        Segments the input frame using a naive Bayes classifier (trained on HSV
        values). This function requires that the Bayes classifier is already available.
        Use the `trainer.classifier.trainBayesClassifier()` function to create it.

        Parameters
        ----------        
        frame: array
            The array containing the colour image data. The input is interpreted
            as a BGR or HSV (see `hsv` argument) 3-channel image. If `frame`
            hasn't got 3 channels, this function fails and returns False.
        hsv (Optional): bool
            If True, the input is interpreted as an HSV 3-channel image. Otherwise,
            it is interpreted as BGR. Default: False

        Returns
        -------
        mask: array
            The final mask image array (2 channels, binary). 255 represents brick
            pixels, 0 for the mortar.
        """

        # check that there is a bayes classifier available
        if self.bayes_clf is None:
            self.logger.error(('The Bayes classifier is not available. Cannot '
                               'run HSV segmentation.'))
            return None
        # check that the input frame has 3 channels (BGR)
        if len(frame.shape) != 3 or frame.shape[2] != 3: 
            self.logger.error('The frame input does not contain 3 channels.')
            return None

        # convert input to HSV if it is not already that
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV) if not hsv else frame
        # make predictions for each pixel with naive bayes
        pred = self.bayes_clf.predict(frame_hsv.reshape(-1,3))
        pred *= 255
        pred = pred.reshape(frame.shape[:2]).astype(np.uint8)

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        closing = cv.morphologyEx(pred, cv.MORPH_CLOSE, kernel)

        # generate final mask based on connected component analysis
        mask = vision.imgproc.connectedComponentsBasedFilter(pred)
        
        return mask


    def locateBricksHsv(self, frame, hsv=False):
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
        hsv (Optional): bool
            If True, the input is interpreted as an HSV 3-channel image. Otherwise,
            it is interpreted as BGR. Default: False

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
        
        # convert input to HSV if it is not already that
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV) if not hsv else frame
        # hsv thresholding
        frame_threshold = cv.inRange(frame_hsv, self.lowerb, self.upperb)

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        closing = cv.morphologyEx(frame_threshold, cv.MORPH_CLOSE, kernel)
        
        # generate final mask based on connected component analysis
        mask = vision.imgproc.connectedComponentsBasedFilter(closing)

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

        if self.debug:
            cv.imshow('Histogram based segmentation',np.vstack([frame, image_row, image_col]))
            cv.imshow('Mask Histogram',mask)

        return mask


    def locateBricksDepth(self, frame):
        mask_plane, mask_ransac = vision.imgproc.separateDepthPlanes(frame)

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        closing = cv.morphologyEx(mask_plane, cv.MORPH_CLOSE, kernel)
        
        # generate final mask based on connected component analysis
        mask = vision.imgproc.connectedComponentsBasedFilter(closing)

        if self.debug:
            cv.imshow('Depth planes based segmentation', mask)

        return mask


    def __processFrames(self, frames):
        frame_colour = frames['colour'].copy()
        frame_depth = frames['depth'].copy()

        mask_hsv = self.locateBricksHsv(frame_colour)
        mask_bayes = self.locateBricksBayes(frame_colour)
        mask_hist = self.locateBricksHistogram(frame_colour, 0)
        mask_depth = self.locateBricksDepth(frame_depth)

        #TODO: implement a better way to combine the masks maybe based on propabilities
        # final mask
        mask = cv.bitwise_or(mask_hsv,mask_bayes)

        masks = [mask_hsv, mask_bayes, mask_hist]
        masks = [mask_.astype(np.float32) for mask_ in masks]
        mask2 = np.zeros(masks[0].shape)
        for mask_ in masks:
            mask2 += mask_
        mask2 *= 255/mask2.max() # division by 0
        mask2 = mask2.astype(np.uint8)
        if self.debug:
            cv.imshow('mask2',mask2)

        if self.debug:
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