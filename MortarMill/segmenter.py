# builtins
import logging

# 3rd party
import cv2 as cv
import numpy as np
from scipy.stats import kurtosis
from sklearn import preprocessing
from skimage.segmentation import random_walker
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
    svm_clf: sklearn.svm._classes.SVC
        SVM classifier generated during training of the classifier.
    dt_clf: sklearn.svm._classes.SVC
        Decision tree classifier generated during training of the classifier.
    randomforest_clf: sklearn.svm._classes.SVC
        Random forest classifier generated during training of the classifier.
    unet: ml.unet.unet.UNet
        UNet classifier loaded from trained model (*.tar file).
    weights: array
        Array of softmax weights used during the combination of the masks.
    mask_threshold: int
        A threshold used to binarize the final mask.
    
    Methods
    -------
    locateBricksHsv(frame, hsv=False)
        Segments the input frame based on HSV threshold values. This function
        requires that the lower and upper HSV thresholds are already available.
        Use the `vision.imgproc.calibrateHsvThresholds()` function to calibrate them.
    locateBricksDepth(frame)
        Segments the input frame by fitting a plane on the depth map. Works
        under the assumption that most of the pixel represent the surface of the
        brick in the input.
    locateBricksSvm(frame, hsv=False, scale=0.25)
        Segments the input frame using a trained SVM. This function
        requires that the SVM classifier is already available. Use the
        `ml.classifier.trainClassifiers()` function to create it.
    locateBricksDecisionTree(self, frame, depth, hsv=False)
        Segments the input frame using a trained DT. This function
        requires that the DT classifier is already available. Use the
        `ml.classifier.trainClassifiers()` function to create it.
    locateBricksRandomForest(self, frame, depth, hsv=False)
        Segments the input frame using a trained RF. This function
        requires that the RF classifier is already available. Use the
        `ml.classifier.trainClassifiers()` function to create it.
    locateBricksUnet(frame)
        Segments the input frame using a trained UNet model. This function
        requires that the UNet model is already available (trained and loaded).
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
            # load the svm classifier
            self.svm_clf = ml.classifier.loadClassifier(config.get('svm'))
            # load the decision tree classifier
            self.dt_clf = ml.classifier.loadClassifier(config.get('dt'))
            # load the random forest classifier
            self.randomforest_clf = ml.classifier.loadClassifier(config.get('randomforest'))
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

            # classifier weights for combination
            self.weights = []
            # random forest
            if self.randomforest_clf is not None:
                self.weights.append(0.93)
            # decision tree
            if self.dt_clf is not None:
                self.weights.append(0.92)
            # unet
            if self.unet is not None:
                self.weights.append(0.92)
            # support vector machine
            if self.svm_clf is not None:
                self.weights.append(0.92)
            # best-fit plane
            self.weights.append(0.76)
            # hsv segmentation
            if self.lowerb is not None and self.upperb is not None:
                self.weights.append(0.64)
            # random walker
            #self.weights.append(0.63)

            # softmax for the weights
            self.weights = np.exp(self.weights)/np.sum(np.exp(self.weights))

            # threshold for the combination of masks
            self.mask_threshold = config.getint('mask_threshold', 140)


    def __call__(self, frames):
        return self.__processFrames(frames)


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


    def locateBricksRandomWalk(self, frame):
        return None

        # check that the input frame has 3 channels (BGR)
        if len(frame.shape) != 3 or frame.shape[2] != 3: 
            self.logger.error('The frame input does not contain 3 channels.')
            return None

        CF, TF = vision.imgproc.calculateColorAndTextureFeatures(frame.copy())
        img = np.dstack([CF,TF])

        fcm = vision.FCM(n_clusters=2,m=1.3,max_iter=1500,error=1e-7)
        fcm.fit(img.reshape(-1,4))


        def getMarkers(fcm, img, features, ratio=0.05):
            u = fcm.predict(features)

            # outputs
            fcm_labels = u.argmax(axis=1)
            unique, counts = np.unique(fcm_labels, return_counts=True)

            fcm_labels = np.ones((features.shape[0],)) * 0.5
            keys = unique
            for i in unique:
                args = u.argpartition(-int(counts[i]*ratio),axis=0)[-int(counts[i]*ratio):]
                fcm_labels[args[:,i]] = i

            fcm_labels[fcm_labels==0] = 2
            fcm_labels[fcm_labels==0.5] = 0
            fcm_labels = fcm_labels.reshape(img.shape[:2])

            return fcm_labels

        markers = getMarkers(fcm,img,img.reshape(-1,4))

        # Run random walker algorithm
        labels = random_walker(img, markers, beta=10, mode='bf', multichannel=True)
        pred = labels-1
        pred *= 255
        pred = pred.astype(np.uint8)

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        closing = cv.morphologyEx(pred, cv.MORPH_CLOSE, kernel)
        
        # generate final mask based on connected component analysis
        mask = vision.imgproc.connectedComponentsBasedFilter(closing)

        return mask


    def locateBricksSvm(self, frame, depth, hsv=False, scale=0.25):
        """ Segments the input frame using a trained SVM. This function
        requires that the SVM classifier is already available. Use the
        `ml.classifier.trainClassifiers()` function to create it.

        Parameters
        ----------        
        frame: array
            The array containing the colour image data. The input is interpreted
            as a BGR 3-channel image. If `frame` hasn't got 3 channels, this
            function fails and returns None.
        depth: array
            The array containing the depth data. The input is interpreted
            as a single channel image. If `frame` hasn't got 1 channel, this
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
        # check that the depth frame has 1 channel
        if len(depth.shape) != 2:
            self.logger.error('The depth input does not contain a single channel.')
            return None
        
        # convert and scale colour input
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV) if not hsv else frame
        frame_hsv = cv.resize(frame_hsv, (int(frame.shape[1]*scale),int(frame.shape[0]*scale)))
        frame_hsv = vision.imgproc.normalise(frame_hsv)
        # scale depth input
        depth_ = cv.resize(depth, (int(depth.shape[1]*scale),int(depth.shape[0]*scale)))
        depth_ = vision.imgproc.normalise(depth_,mode='depth')
        # combine colour and depth information
        features = np.hstack([frame_hsv.reshape(-1,3),depth_.reshape(-1,1)])

        # perform SVM based segmentation
        pred = self.svm_clf.predict(features) * 255
        pred = pred.reshape(frame_hsv.shape[:2]).astype(np.uint8)
        pred = cv.resize(pred, (frame.shape[1], frame.shape[0]))

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        closing = cv.morphologyEx(pred, cv.MORPH_CLOSE, kernel)
        
        # generate final mask based on connected component analysis
        mask = vision.imgproc.connectedComponentsBasedFilter(closing)

        return mask


    def locateBricksDecisionTree(self, frame, depth, hsv=False):
        """ Segments the input frame using a trained decision tree classifier.
        This function requires that the DT classifier is already available. Use the
        `ml.classifier.trainClassifiers()` function to create it.

        Parameters
        ----------        
        frame: array
            The array containing the colour image data. The input is interpreted
            as a BGR 3-channel image. If `frame` hasn't got 3 channels, this
            function fails and returns None.
        depth: array
            The array containing the depth data. The input is interpreted
            as a single channel image. If `frame` hasn't got 1 channel, this
            function fails and returns None.
        hsv: bool (default: False)
            If True, the input is interpreted as an HSV 3-channel image. Otherwise,
            it is interpreted as BGR. Default: False

        Returns
        -------
        mask: array
            The final mask image array (1 channel, binary). 255 represents brick
            pixels, 0 for the mortar.
        """

        # check if the DT classifier is available
        if self.dt_clf is None:
            self.logger.error('The DT classifier is None; cannot do segmentation.')
            return None
        # check that the input frame has 3 channels (BGR)
        if len(frame.shape) != 3 or frame.shape[2] != 3: 
            self.logger.error('The frame input does not contain 3 channels.')
            return None
        # check that the depth frame has 1 channel
        if len(depth.shape) != 2:
            self.logger.error('The depth input does not contain a single channel.')
            return None

        # convert and scale colour input
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV) if not hsv else frame
        frame_hsv = vision.imgproc.normalise(frame_hsv)
        # scale depth input
        depth_ = vision.imgproc.normalise(depth,mode='depth')
        # combine colour and depth information
        features = np.hstack([frame_hsv.reshape(-1,3),depth_.reshape(-1,1)])

        # predict with DT
        pred = self.dt_clf.predict(features) * 255
        pred = pred.reshape(frame_hsv.shape[:2]).astype(np.uint8)

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        closing = cv.morphologyEx(pred, cv.MORPH_CLOSE, kernel)
        
        # generate final mask based on connected component analysis
        mask = vision.imgproc.connectedComponentsBasedFilter(closing)

        return mask


    def locateBricksRandomForest(self, frame, depth, hsv=False):
        """ Segments the input frame using a trained random forest classifier.
        This function requires that the RF classifier is already available. Use the
        `ml.classifier.trainClassifiers()` function to create it.

        Parameters
        ----------        
        frame: array
            The array containing the colour image data. The input is interpreted
            as a BGR 3-channel image. If `frame` hasn't got 3 channels, this
            function fails and returns None.
        depth: array
            The array containing the depth data. The input is interpreted
            as a single channel image. If `frame` hasn't got 1 channel, this
            function fails and returns None.
        hsv: bool (default: False)
            If True, the input is interpreted as an HSV 3-channel image. Otherwise,
            it is interpreted as BGR. Default: False

        Returns
        -------
        mask: array
            The final mask image array (1 channel, binary). 255 represents brick
            pixels, 0 for the mortar.
        """

        # check if the RF classifier is available
        if self.randomforest_clf is None:
            self.logger.error('The RF classifier is None; cannot do segmentation.')
            return None
        # check that the input frame has 3 channels (BGR)
        if len(frame.shape) != 3 or frame.shape[2] != 3: 
            self.logger.error('The frame input does not contain 3 channels.')
            return None
        # check that the depth frame has 1 channel
        if len(depth.shape) != 2:
            self.logger.error('The depth input does not contain a single channel.')
            return None
        
        # convert and scale colour input
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV) if not hsv else frame
        frame_hsv = vision.imgproc.normalise(frame_hsv)
        # scale depth input
        depth_ = vision.imgproc.normalise(depth,mode='depth')
        # combine colour and depth information
        features = np.hstack([frame_hsv.reshape(-1,3),depth_.reshape(-1,1)])

        # predict with RF
        pred = self.dt_clf.predict(features) * 255
        pred = pred.reshape(frame_hsv.shape[:2]).astype(np.uint8)

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        closing = cv.morphologyEx(pred, cv.MORPH_CLOSE, kernel)
        
        # generate final mask based on connected component analysis
        mask = vision.imgproc.connectedComponentsBasedFilter(closing)

        return mask


    def locateBricksUnet(self, frame, depth):
        """ Segments the input frame using a trained UNet model. This function
        requires that the UNet model is already available (trained and loaded).

        Parameters
        ----------        
        frame: array
            The array containing the colour image data. The input is interpreted
            as a BGR 3-channel image. If `frame` hasn't got 3 channels, this
            function fails and returns None.
        depth: array
            The array containing the depth data. The input is interpreted
            as a single channel image. If `frame` hasn't got 1 channel, this
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
        if len(depth.shape) != 2:
            self.logger.error('The depth input does not contain a single channel.')
            return None
        
        # perform segmentation with UNet
        frame_ = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pred = self.unet.predict(Image.fromarray(frame_),depth)

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        closing = cv.morphologyEx(pred, cv.MORPH_CLOSE, kernel)
        
        # generate final mask based on connected component analysis
        mask = vision.imgproc.connectedComponentsBasedFilter(closing)

        return mask


    def __processFrames(self, frames):
        frame_colour = frames['colour'].copy()
        frame_depth = frames['depth'].copy()

        # perform segmentation with selected algorithms
        mask_rf = self.locateBricksRandomForest(frame_colour, frame_depth)
        mask_dt = self.locateBricksDecisionTree(frame_colour, frame_depth)
        mask_unet = self.locateBricksUnet(frame_colour, frame_depth)
        mask_svm = self.locateBricksSvm(frame_colour, frame_depth)
        mask_depth = self.locateBricksDepth(frame_depth)
        mask_hsv = self.locateBricksHsv(frame_colour)
        mask_rw = self.locateBricksRandomWalk(frame_colour)

        # weighted sum of masks
        masks = [mask_rf, mask_dt, mask_unet, mask_svm, mask_depth, mask_hsv, mask_rw]
        masks = [mask_.astype(np.float64) for mask_ in masks if mask_ is not None]
        mask_combined = np.zeros(masks[0].shape)
        for i, mask_ in enumerate(masks):
            mask_combined += mask_ * self.weights[i]
        mask_combined = np.uint8(np.round(mask_combined))

        # final mask using threshold
        mask = np.where(mask_combined >= self.mask_threshold, 255, 0).astype(np.uint8)

        return mask