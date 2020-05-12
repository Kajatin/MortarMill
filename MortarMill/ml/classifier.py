import os
import pickle
import logging

import cv2 as cv
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#from . import dataset
import dataset
import vision


logger = logging.getLogger(__name__)


def trainClassifiers(clfs, names, path):
    """ Trains the provided scikit classifiers `clfs`.

    Parameters
    ----------
    clf: array
        Array of scikit classifiers to be trained.
    names: array
        Array of the names of the scikit classifiers.
    path: string, array
        String that specifies the path to the training data. If an array, it should
        be an array of strings, where each string specifies the path to the training
        file.
    """

    # load the sample images and create the dataset
    data = dataset.loadDatasetPath(path)
    # check if the dataset could be created
    if data is None:
        logger.error('Cannot train classifier since the dataset could not be created.')
        return None

    # create the training data from the given samples
    X, y = dataset.createTrainingData(data, 0.05, mode='labelled', feature='hsv')
    # scale the training data to normal distribution
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    logger.info(('Created dataset to train classifier on. Train size: {}, test '
                 'size: {}.').format(X_train.shape[0],X_test.shape[0]))
    
    # loop over the classifiers and train each of them
    for name, clf in zip(names, clfs):
        logger.info('Training {} classifier.'.format(name))
        # train the classifier
        clf.fit(X_train, y_train)
        # test clf
        score = clf.score(X_test, y_test)
        logger.info('Classifier trained with score: {}.'.format(score))
        # save the trained classifier
        saveClassifier([clf,scaler], f'{name}.pickle')


def saveClassifier(clf, file, path='models/'):
    """ Saves a classifier and its scaler as a pickle file to the given path and
    file name.

    Parameters
    ----------
    clf: array
        Array that contains the classifier and the scaler to be saved. There is
        no check as to whether `clf` is actually a classifier or not. Any object
        provided will be saved as a pickle file.
    file: string
        Name of the pickle file (with `.pickle` file extension).
    path: string (default: 'models/')
        String that specifies the path to the pickle file.
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
    ret: array
        Returns the loaded classifier (and scaler) as an array normally.
        Otherwise returns None.
    """

    if file is None:
        logger.error('Cannot load classifier because filename is None.')
        return None

    if os.path.isfile(os.path.join(path,file)):
        logger.info('Loading classifier from {}.'.format(os.path.join(path,file)))
        return pickle.load(open(os.path.join(path,file), 'rb'))
    
    logger.error('Cannot load classifier from {}.'.format(os.path.join(path,file)))
    return None


if __name__ == '__main__':
    names = ['knn', 'linear_svm', 'rbf_svm', 'decision_tree', 'random_forest',
             'adaboost', 'naive_bayes', 'qda']

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]


    # train the classifiers
    trainClassifiers(classifiers, names, 'samples/train/')

    # load the SVM classifier if it exists
    clf, scaler = loadClassifier('rbf_svm.pickle')
    
    # test the classifier on an image
    image = dataset.getRandomTestImage()
    predictions = clf.predict(scaler.transform(cv.cvtColor(image, cv.COLOR_BGR2HSV).reshape(-1,3)))
    predictions = predictions.reshape(image.shape[:2]).astype(np.uint8)
    predictions *= 255
    
    cv.imshow('predictions',predictions)
    cv.imshow('original',image)
    key = cv.waitKey(0)