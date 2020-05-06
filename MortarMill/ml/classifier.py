import os
import pickle
import logging

import cv2 as cv
import numpy as np
from sklearn import svm, naive_bayes, preprocessing

from . import dataset
import vision

logger = logging.getLogger(__name__)


def trainSvmClassifier(path, unsupervised=True):
    # load the sample images and create the dataset
    data = ml.dataset.loadDatasetPath(path)
    # check if the dataset could be created
    if data is None:
        logger.error('Cannot train SVM since the dataset could not be created.')
        return None

    # create the training data from the given samples
    X, y = ml.dataset.createTrainingData(data, unsupervised)
    # scale the training data to normal distribution
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    print('Training data size: {}'.format(X.shape[0]))
    
    # train an SVM
    clf = svm.SVC(C=1)
    clf.fit(X, y)
    print(clf.n_support_)

    return clf, scaler


def trainBayesClassifier(arg):
    if isinstance(arg, np.ndarray):
        # convert to HSV
        frame_hsv = cv.cvtColor(arg,cv.COLOR_BGR2HSV)
        
        # manually select ROI of brick to generate training data
        r = cv.selectROI('Select brick area',arg)
        cv.destroyWindow('Select brick area')
        # crop image
        frame_cropped = frame_hsv[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        brick = frame_cropped.reshape(-1,3)
        brick_y = np.ones(brick.shape[0])

        # manually select ROI of mortar to generate training data
        r = cv.selectROI('Select mortar area',arg)
        cv.destroyWindow('Select mortar area')
        # crop image
        frame_cropped = frame_hsv[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        mortar = frame_cropped.reshape(-1,3)
        mortar_y = np.zeros(mortar.shape[0])

        X = np.vstack([brick,mortar])
        y = np.hstack([brick_y,mortar_y])
    
    else:
        # load the sample images and create the dataset
        data = ml.dataset.loadDatasetPath(arg)
        # check if the dataset could be created
        if data is None:
            logger.error(('Cannot train Bayes classifier since the dataset could'
                          ' not be created.'))
            return None

        # create the training data from the given samples
        X, y = ml.dataset.createHsvTrainingData(data)

    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    # train Bayes classifier
    clf = naive_bayes.GaussianNB()
    clf.fit(X, y)

    return clf, scaler


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
        Name of the pickle file (with `.pickle` file extension). If None, returns
        None.
    path: string
        String that specifies the path to the pickle file.

    Returns
    -------
    ret: bool
        Returns the loaded classifier nominally. Otherwise returns None.
    """

    if file is None:
        logger.error('Cannot load classifier because filename is None.')
        return None

    if os.path.isfile(os.path.join(path,file)):
        logger.info('Loading classifier from {}.'.format(os.path.join(path,file)))
        return pickle.load(open(os.path.join(path,file), 'rb'))
    
    logger.error('Cannot load classifier from {}.'.format(os.path.join(path,file)))
    return None


def assignLabelsUnsupervised(image, features, features_t=None, ratio=0.05):
    #TODO: remove image input, not needed

    if features_t is None:
        features_t = features

    # fuzzy c-means
    # fit the fuzzy-c-means
    fcm = vision.FCM(n_clusters=2,m=1.3,max_iter=1500,error=1e-7,random_state=np.random.randint(10000))
    fcm.fit(features)

    # outputs
    fcm_centers = fcm.centers
    fcm_labels  = fcm.u.argmax(axis=1)
    unique, counts = np.unique(fcm_labels, return_counts=True)
    print(fcm_centers)
    print(unique, counts)

    fcm_labels = np.ones((features.shape[0],)) * 0.5
    training = {}
    keys = unique
    for i in unique:
        args = fcm.u.argpartition(-int(counts[i]*ratio),axis=0)[-int(counts[i]*ratio):]
        fcm_labels[args[:,i]] = i
        training[keys[i]] = features_t[args[:,i],:]

    fcm_labels = fcm_labels.reshape(image.shape[:2])
    fcm_labels = cv.normalize(fcm_labels, None, 0, 255, cv.NORM_MINMAX)
    fcm_labels = fcm_labels.astype(np.uint8)
    cv.imshow('fcm_labels',fcm_labels)

    fcm_labels  = fcm.u.argmax(axis=1)
    fcm_labels = fcm_labels.reshape(image.shape[:2])
    fcm_labels = cv.normalize(fcm_labels, None, 0, 255, cv.NORM_MINMAX)
    fcm_labels = fcm_labels.astype(np.uint8)
    cv.imshow('fcm_labels_orig',fcm_labels)

    return training


if __name__ == '__main__':
    # load the SVM classifier if it exists
    #clf = loadClassifier('bayes_clf.pickle')
    clf = None

    if clf is None:
        # otherwise train a new SVM classifier
        #clf, scaler = trainSvmClassifier('samples/train/', False)
        #clf = trainSvmClassifier(['samples/RAW/brick_zoom_5.jpg'])
        #clf = trainSvmClassifier(['samples/RAW/brick_5.jpg'])
        #clf = trainSvmClassifier(['samples/flower2.jpg'])
        clf, scaler = trainBayesClassifier('samples/train/')
        # and save it
        #saveClassifier([clf,scaler], 'svm_rgb.pickle')
        #saveClassifier([clf,scaler], 'bayes_clf.pickle')

    # test the classifier on an image
    image = ml.dataset.getRandomTestImage()
    #predictions = clf.predict(scaler.transform(image.reshape(-1,3)))
    predictions = clf.predict(scaler.transform(cv.cvtColor(image, cv.COLOR_BGR2HSV).reshape(-1,3)))
    #CF, TF = vision.imgproc.calculateColorAndTextureFeatures(image.copy())
    #features = np.hstack((CF.reshape(-1,3),TF.reshape(-1,1)))
    #features = CF.reshape(-1,3)
    #predictions = clf.predict(features)

    predictions = predictions.reshape(image.shape[:2])
    #predictions = cv.normalize(predictions, None, 0, 255, cv.NORM_MINMAX)
    predictions *= 255
    predictions = predictions.astype(np.uint8)
    
    cv.imshow('predictions',predictions)
    cv.imshow('original',image)
    key = cv.waitKey(0)



    ## load the SVM classifier if it exists
    #clf2 = loadClassifier('svm_hsv.pickle')
    #clf3 = loadClassifier('svm_rgb.pickle')

    ## test the classifier on an image
    #image = ml.dataset.getRandomTestImage()

    #predictions2 = clf2.predict(cv.cvtColor(image, cv.COLOR_BGR2HSV).reshape(-1,3))
    #predictions3 = clf3.predict(image.reshape(-1,3))

    #predictions2 = predictions2.reshape(image.shape[:2])
    #predictions3 = predictions3.reshape(image.shape[:2])

    #predictions2 *= 255
    #predictions3 *= 255
    
    #predictions2 = predictions2.astype(np.uint8)
    #predictions3 = predictions3.astype(np.uint8)
    
    #cv.imshow('predictions2',predictions2)
    #cv.imshow('predictions3',predictions3)
    #cv.imshow('original',image)
    #key = cv.waitKey(0)