import os
import pickle
import logging

import cv2 as cv
import numpy as np
from sklearn import svm, naive_bayes

import trainer.dataset

logger = logging.getLogger(__name__)


def trainSvmClassifier(path, unsupervised=True):
    # load the sample images and create the dataset
    data = trainer.dataset.loadDatasetPath(path)
    # check if the dataset could be created
    if data is None:
        logger.error('Cannot train SVM since the dataset could not be created.')
        return None

    # create the training data from the given samples
    X, y = trainer.dataset.createTrainingData(data, unsupervised)

    print('Training data size: {}'.format(X.shape[0]))
    
    # train an SVM
    clf = svm.SVC()
    clf.fit(X, y)

    return clf


def trainBayesClassifier(path):
    # load the sample images and create the dataset
    data = trainer.dataset.loadDatasetPath(path)
    # check if the dataset could be created
    if data is None:
        logger.error(('Cannot train Bayes classifier since the dataset could'
                      ' not be created.'))
        return None

    # create the training data from the given samples
    X, y = trainer.dataset.createHsvTrainingData(data)

    # train Bayes classifier
    clf = naive_bayes.GaussianNB()
    clf.fit(X, y)

    return clf


def saveClassifier(clf, file, path='models/'):
    """ Saves a classifier as a pickle file to the given path and file name.

    Parameters
    ----------
    clf: object
        Classifier to be saved. There is no check as to whether `clf` is actually
        a classifier or not. Any object provided will be saved as a pickle file.
    file: string
        Name of the pickle file (with `.pickle` file extension).
    path: string
        String that specifies the path to the pickle file.

    Returns
    -------
    ret: bool
        Returns the loaded classifier nominally. Otherwise returns False.
    """

    assert isinstance(file, str)

    # check if the file extension is provided in the file name
    if '.pickle' not in file:
        logger.debug('Adding `.pickle` file extension to the provided file name.')
        file += '.pickle'

    # create the save directory if it does not exist
    if not os.path.isdir(path):
        logger.debug('Creating save directory {}.'.format(path))
        os.makedirs(path)

    # check if the given file already exists and issue warning that it will be overwritten
    save_path = os.path.join(path,file)
    if os.path.isfile(save_path):
        logger.warning('{} already exists. It will be overwritten.'.format(save_path))

    pickle.dump(clf, open(save_path, 'wb'))


def loadClassifier(file, path='models/'):
    """ Loads a classifier saved as a pickle file from the given path and file name.

    Parameters
    ----------
    file: string
        Name of the pickle file (with `.pickle` file extension).
    path: string
        String that specifies the path to the pickle file.

    Returns
    -------
    ret: bool
        Returns the loaded classifier nominally. Otherwise returns None.
    """

    if os.path.isfile(os.path.join(path,file)):
        logger.info('Loading classifier from {}.'.format(os.path.join(path,file)))
        return pickle.load(open(os.path.join(path,file), 'rb'))
    
    logger.error('Cannot load classifier from {}.'.format(os.path.join(path,file)))
    return None


if __name__ == '__main__':
    # load the SVM classifier if it exists
    clf = loadClassifier('bayes_clf.pickle')
    #clf = None

    if clf is None:
        # otherwise train a new SVM classifier
        #clf = trainSvmClassifier('samples/train/', False)
        #clf = trainSvmClassifier(['samples/RAW/brick_zoom_5.jpg'])
        clf = trainBayesClassifier('samples/train/')
        # and save it
        saveClassifier(clf, 'bayes_clf.pickle')

    # test the classifier on an image
    image = trainer.dataset.getRandomTestImage()
    #predictions = clf.predict(image.reshape(-1,3))
    predictions = clf.predict(cv.cvtColor(image, cv.COLOR_BGR2HSV).reshape(-1,3))

    predictions = predictions.reshape(image.shape[:2])
    predictions = cv.normalize(predictions, None, 0, 255, cv.NORM_MINMAX)
    predictions = predictions.astype(np.uint8)
    cv.imshow('predictions',predictions)

    cv.imshow('original',image)

    key = cv.waitKey(0)