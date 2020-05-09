# builtins
import logging

# 3rd party
import cv2 as cv
import numpy as np
from scipy.stats import kurtosis
from sklearn import preprocessing
import torch
from PIL import Image

# custom
import vision
import ml


class Singleton(type):
    """ Metaclass used to create a singleton class instance """

    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Segmenter(metaclass=Singleton):
    """ Performs the image segmentation of the input image. Outputs a binary mask
    of the input frame, where brick pixels are represented with a value of 255 and
    the mortar with 0.

    Parameters
    ----------
    config: configparser.SectionProxy
        Configuration file with information about the classifiers to load. If None,
        the segmenter will not function.

    Attributes
    ----------
    debug: bool
        If True, the class is operated in debug mode with verbose outputs. Otherwise,
        the class only outputs the final mask.
    lowerb, upperb: tuple, tuple
        Two tuples that hold the lower and upper HSV segmentation range values.
    bayes_clf, bayes_scaler: sklearn.naive_bayes.GaussianNB, sklearn.preprocessing._data.StandardScaler
        Bayes classifier and scaler generated during training of the classifier.
    svm_clf, svm_scaler: sklearn.svm._classes.SVC, sklearn.preprocessing._data.StandardScaler
        SVM classifier and scaler generated during training of the classifier.
    unet: ml.unet.unet.UNet
        UNet classifier loaded from trained model (*.tar file).
    
    Methods
    -------
    locateBricksBayes(frame, hsv=False)
        Segments the input frame using a naive Bayes classifier (trained on HSV
        values). This function requires that the Bayes classifier is already available.
        Use the `ml.classifier.trainBayesClassifier()` function to create it.
    locateBricksHsv(frame, hsv=False)
        Segments the input frame based on HSV threshold values. This function
        requires that the lower and upper HSV thresholds are already available.
        Use the `vision.imgproc.calibrateHsvThresholds()` function to calibrate them.
    locateBricksSvm(frame, hsv=False, scale=0.25)
        Segments the input frame using a trained SVM. This function
        requires that the SVM classifier is already available. Use the
        `ml.classifier.trainSvmClassifier()` function to create it.
    locateBricksUnet(frame)
        Segments the input frame using a trained UNet model. This function
        requires that the UNet model is already available (trained and loaded).
    locateBricksDepth(frame)
        Segments the input frame by fitting a plane on the depth map. Works
        under the assumption that most of the pixel represent the surface of the
        brick in the input.
    """

    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.logger.debug('Creating Segmenter instance.')

        # parse the config file if it is not None
        if config is None:
            self.logger.warning(('No config is provided to Segmenter. Some '
                                 'parameters might not be available, and this '
                                 'module might not work correctly.'))
        else:
            self.logger.debug('Parsing config for Segmenter. {}'.format(config))
            # set the debug mode from config file
            self.debug = config.getboolean('debug', False)
            # load the HSV segmentation values
            lowerb = (config.getint('min_h'),config.getint('min_s'),config.getint('min_v'))
            upperb = (config.getint('max_h'),config.getint('max_s'),config.getint('max_v'))
            self.lowerb = lowerb if all(lowerb) else None
            self.upperb = upperb if all(upperb) else None
            # load bayes classifier
            bayes_clf = ml.classifier.loadClassifier(config.get('bayes'))
            self.bayes_clf, self.bayes_scaler = bayes_clf if bayes_clf is not None else (None, None)
            # load the svm classifier
            svm_clf = ml.classifier.loadClassifier(config.get('svm'))
            self.svm_clf, self.svm_scaler = svm_clf if svm_clf is not None else (None, None)
            # load the UNet model
            unet_config = config.get('unet')
            if unet_config is None:
                self.logger.warning('No UNet model is provided in the configuration file.')
                self.unet = None
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.unet = ml.unet.UNet()
                self.unet.to(device=device)
                checkpoint = torch.load(unet_config, map_location=device)
                self.unet.load_state_dict(checkpoint['model_state_dict'])


    def __call__(self, frames):
        self.__processFrames(frames)


    def locateBricksBayes(self, frame, hsv=False):
        """ Segments the input frame using a naive Bayes classifier (trained on HSV
        values). This function requires that the Bayes classifier is already available.
        Use the `ml.classifier.trainBayesClassifier()` function to create it.

        Parameters
        ----------        
        frame: array
            The array containing the colour image data. The input is interpreted
            as a BGR or HSV (see `hsv` argument) 3-channel image. If `frame`
            hasn't got 3 channels, this function fails and returns None.
        hsv (Optional): bool
            If True, the input is interpreted as an HSV 3-channel image. Otherwise,
            it is interpreted as BGR. Default: False

        Returns
        -------
        mask: array
            The final mask image array (1 channel, binary). 255 represents brick
            pixels, 0 for the mortar.
        """

        # check that there is a bayes classifier available
        if self.bayes_clf is None:
            self.logger.error(('The Bayes classifier is not available. Cannot '
                               'run segmentation.'))
            return None
        # check that the input frame has 3 channels (BGR)
        if len(frame.shape) != 3 or frame.shape[2] != 3: 
            self.logger.error('The frame input does not contain 3 channels.')
            return None

        # convert input to HSV if necessary
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV) if not hsv else frame
        # make predictions for each pixel with naive bayes
        pred = self.bayes_clf.predict(self.bayes_scaler.transform(frame_hsv.reshape(-1,3)))
        pred *= 255
        pred = pred.reshape(frame.shape[:2]).astype(np.uint8)

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        closing = cv.morphologyEx(pred, cv.MORPH_CLOSE, kernel)

        # generate final mask based on connected component analysis
        mask = vision.imgproc.connectedComponentsBasedFilter(pred)
        
        return mask


    def locateBricksHsv(self, frame, hsv=False):
        """ Segments the input frame based on HSV threshold values. This function
        requires that the lower and upper HSV thresholds are already available.
        Use the `vision.imgproc.calibrateHsvThresholds()` function to calibrate them.

        Parameters
        ----------        
        frame: array
            The array containing the colour image data. The input is interpreted
            as a BGR 3-channel image. If `frame` hasn't got 3 channels, this
            function fails and returns None.
        hsv (Optional): bool
            If True, the input is interpreted as an HSV 3-channel image. Otherwise,
            it is interpreted as BGR. Default: False

        Returns
        -------
        mask: array
            The final mask image array (1 channel, binary). 255 represents brick
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


    def locateBricksSvm(self, frame, hsv=False, scale=0.25):
        """ Segments the input frame using a trained SVM. This function
        requires that the SVM classifier is already available. Use the
        `ml.classifier.trainSvmClassifier()` function to create it.

        Parameters
        ----------        
        frame: array
            The array containing the colour image data. The input is interpreted
            as a BGR 3-channel image. If `frame` hasn't got 3 channels, this
            function fails and returns None.
        hsv: bool (default: False)
            If True, the input is interpreted as an HSV 3-channel image. Otherwise,
            it is interpreted as BGR. Default: False
        scale: float (default: 0.25)
            Used to scale the `frame` down to speed up the prediction process.

        Returns
        -------
        mask: array
            The final mask image array (1 channel, binary). 255 represents brick
            pixels, 0 for the mortar.
        """

        # check if the SVM classifier is available
        if self.svm_clf is None:
            self.logger.error('The SVM classifier is None; cannot do segmentation.')
            return None
        # check that the input frame has 3 channels (BGR)
        if len(frame.shape) != 3 or frame.shape[2] != 3: 
            self.logger.error('The frame input does not contain 3 channels.')
            return None
        # make sure that the scaler is in the range (0,1]
        if 0 >= scale > 1:
            self.logger.error(('The argument `scale` is expected to be in the range'
                              ' (0,1], but it was {}.').format(scale))
            return None
        
        # perform SVM based segmentation
        frame_ = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_ = cv.resize(frame_, (int(frame.shape[1]*scale),int(frame.shape[0]*scale)))
        pred = self.svm_clf.predict(self.svm_scaler.transform(frame_.reshape(-1,3)))
        pred *= 255
        pred = pred.reshape(frame_.shape[:2]).astype(np.uint8)
        pred = cv.resize(pred, (frame.shape[1], frame.shape[0]))

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        closing = cv.morphologyEx(pred, cv.MORPH_CLOSE, kernel)
        
        # generate final mask based on connected component analysis
        mask = vision.imgproc.connectedComponentsBasedFilter(closing)

        return mask


    def locateBricksUnet(self, frame):
        """ Segments the input frame using a trained UNet model. This function
        requires that the UNet model is already available (trained and loaded).

        Parameters
        ----------        
        frame: array
            The array containing the colour image data. The input is interpreted
            as a BGR 3-channel image. If `frame` hasn't got 3 channels, this
            function fails and returns None.

        Returns
        -------
        mask: array
            The final mask image array (1 channel, binary). 255 represents brick
            pixels, 0 for the mortar.
        """

        # check if the SVM classifier is available
        if self.unet is None:
            self.logger.error('The UNet model is None; cannot do segmentation.')
            return None
        # check that the input frame has 3 channels (BGR)
        if len(frame.shape) != 3 or frame.shape[2] != 3: 
            self.logger.error('The frame input does not contain 3 channels.')
            return None
        
        # perform segmentation with UNet
        frame_ = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pred = self.unet.predict(Image.fromarray(frame_))
        pred = np.invert(pred)

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        closing = cv.morphologyEx(pred, cv.MORPH_CLOSE, kernel)
        
        # generate final mask based on connected component analysis
        mask = vision.imgproc.connectedComponentsBasedFilter(closing)

        return mask


    def locateBricksDepth(self, frame):
        """ Segments the input frame by fitting a plane on the depth map. Works
        under the assumption that most of the pixel represent the surface of the
        brick in the input.

        Parameters
        ----------        
        frame: array
            The array containing the depth data. The input is interpreted
            as a single channel image. If `frame` hasn't got 1 channel, this
            function fails and returns None.

        Returns
        -------
        mask: array
            The final mask image array (1 channel, binary). 255 represents brick
            pixels, 0 for the mortar.
        """

        # check that the input frame has 1 channel
        if len(frame.shape) != 2:
            self.logger.error('The frame input does not contain a single channel.')
            return None

        # fit a plane on the input
        mask = vision.imgproc.separateDepthPlanes(frame)

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        
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


    def __processFrames(self, frames):
        frame_colour = frames['colour'].copy()
        frame_depth = frames['depth'].copy()

        mask_hsv = self.locateBricksHsv(frame_colour)
        mask_bayes = self.locateBricksBayes(frame_colour)
        mask_svm = self.locateBricksSvm(frame_colour)
        mask_unet = self.locateBricksUnet(frame_colour)
        #mask_depth = self.locateBricksDepth(frame_depth)
        #mask_hist = self.locateBricksHistogram(frame_colour, 0)

        masks = [mask_hsv, mask_bayes, mask_svm, mask_unet]
        #masks = [mask_hsv, mask_bayes, mask_svm, mask_unet, mask_depth]
        masks = [mask_.astype(np.float32) for mask_ in masks if mask_ is not None]
        mask_combined = np.zeros(masks[0].shape)
        for mask_ in masks:
            mask_combined += mask_
        mask_combined *= 255/mask_combined.max() # division by 0
        mask_combined = mask_combined.astype(np.uint8)

        #TODO: implement a better way to combine the masks maybe based on propabilities
        # final mask
        mask = np.where(mask_combined > 130, 255, 0).astype(np.uint8)
        
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
            cv.imshow('Detected bricks',np.vstack((frame_colour_copy,masked_orig)))
            cv.imshow('Combined masks',mask_combined)

            #cv.imwrite('samples/presentation/combined_masked.png',masked_orig)
            #cv.imwrite('samples/presentation/combined.png',frame_colour_copy)
            #cv.imwrite('samples/presentation/combined_masks.png',mask2)


            #from skimage.morphology import skeletonize, medial_axis
            #skeleton = skeletonize(np.invert(mask)/255.0)
            #skeleton = medial_axis(np.invert(mask)/255.0)
            #cv.imshow('inverse mask',np.invert(mask))
            #cv.imshow('Skeleton', (skeleton*255).astype(np.uint8))

            
            #import matplotlib.pyplot as plt
            #data = np.invert(mask)/255.0
            ## Compute the medial axis (skeleton) and the distance transform
            #skel, distance = medial_axis(data, return_distance=True)
            ## Distance to the background for pixels of the skeleton
            #dist_on_skel = distance * skel
            #plt.figure(figsize=(8, 4))
            #plt.subplot(121)
            #plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
            #plt.axis('off')
            #plt.subplot(122)
            #plt.imshow(dist_on_skel, cmap=plt.cm.Spectral, interpolation='nearest')
            #plt.contour(data, [0.5], colors='w')
            #plt.axis('off')
            #plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
            #                    right=1)
            #plt.show()

        return mask