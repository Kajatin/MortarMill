import os
import logging

import cv2 as cv
import numpy as np

import common.preprocessing

logger = logging.getLogger(__name__)


def loadDatasetPath(basepath):
    """ Creates a list of filenames that comprise the dataset.

    Parameters
    ----------
    basepath: string, list
        String that specifies the path to the dataset files.
        If it is a string, it is assumed to be the path to the directory which
        contains the dataset files. If it is a list, it is assumed to contain
        the filenames used to create the dataset.

    Returns
    -------
    dataset: list
        Returns a list of filenames that comprise the dataset. If the input was
        invalid, return None.
    """

    dataset = []

    # if the input is a list, it is assumed to contain the filenames to be used
    # to create the dataset
    if isinstance(basepath, list):
        logger.debug(('Provided path argument is interpreted to contain the '
                     'filenames used to create the dataset. ({})').format(basepath))
        for path in basepath:
            if os.path.isfile(path):
                dataset.append(path)
                logger.debug('Adding {} to dataset.'.format(path))
    # if the input is a string, it is assumed to be the path to a directory which
    # contains all the files used to create the dataset
    elif isinstance(basepath, str):
        logger.debug(('Provided path argument is interpreted to contain the '
                     'directory that contains the dataset files.'
                     ' ({})').format(basepath))
        for entry in os.listdir(basepath):
            path = os.path.join(basepath,entry)
            if os.path.isfile(path):
                dataset.append(path)
                logger.debug('Adding {} to dataset.'.format(path))
    # otherwise, fail to create the dataset
    else:
        logger.error(('Cannot create dataset, because the input ({}) was '
                      'incorrect.'.format(basepath)))

    logger.debug('Returning dataset: {}'.format(dataset))
    # if the dataset is empty return None
    return dataset if len(dataset) > 0 else None


def createTrainingData(dataset, unsupervised=True, ratio=0.05):
    X = []
    y = []

    for path in dataset:
        print(path)
        image = cv.imread(path)
        #image = cv.resize(image, (600, int(image.shape[0] * (600.0/image.shape[1]))))

        #CF, TF = common.preprocessing.calculateColorAndTextureFeatures(image.copy())
        #features = np.hstack((CF.reshape(-1,3),TF.reshape(-1,1)))

        if unsupervised:
            #TODO: implement case where first all CF,TF features are calculated for all images before the FCM is called
            training = common.preprocessing.assignLabelsUnsupervised(image, features, image.reshape(-1,3), ratio)
    
            X.append(np.vstack([training[0],training[1]]))
            y.append(np.hstack([np.repeat(0,len(training[0])),np.repeat(1,len(training[1]))]))
        else:
            #features_ = features
            features_ = image.reshape(-1,3)
            #features_ = cv.cvtColor(image, cv.COLOR_BGR2HSV).reshape(-1,3)

            # select a subset of the data used for training
            selection_features_ = features_[np.random.choice(features_.shape[0],
                                                             int(features_.shape[0]*ratio),
                                                             replace=False), :]

            # determine the label from the file name 0: mortar, 1: brick
            label = int(path.split('_')[-1].split('.')[0])

            X.append(selection_features_)
            y.append(np.repeat(label,selection_features_.shape[0]))

    return np.vstack(X), np.hstack(y)


def createHsvTrainingData(dataset, ratio=0.05):
    X = []
    y = []

    for path in dataset:
        # read image data
        image = cv.imread(path)
        # convert to HSV and reshape to 2D
        features_ = cv.cvtColor(image, cv.COLOR_BGR2HSV).reshape(-1,3)
        # select a subset of the data used for training
        selection_features_ = features_[np.random.choice(features_.shape[0],
                                                         int(features_.shape[0]*ratio),
                                                         replace=False), :]
        # determine the label from the file name 0: mortar, 1: brick
        label = int(path.split('_')[-1].split('.')[0])

        X.append(selection_features_)
        y.append(np.repeat(label,selection_features_.shape[0]))

    return np.vstack(X), np.hstack(y)


#TODO: code this function properly (getRandomTestImage())
def getRandomTestImage():
    image = cv.imread('samples/RAW/brick_zoom_8.jpg')
    return cv.resize(image, (600, int(image.shape[0] * (600.0/image.shape[1]))))