import os

import cv2 as cv
import numpy as np
from sklearn import svm
import pickle

import dataset


def trainClassifier(path):
    # load the sample images from the given directory
    data = dataset.loadDatasetPath(path)
    # create the training data from the given samples
    X, y = dataset.createTrainingData(data)
    
    # train an SVM
    clf = svm.SVC()
    clf.fit(X, y)

    return clf


def saveClassifier(clf, file, path='models/'):
    if not os.path.isdir(path):
        os.makedirs(path)

    if os.path.isfile(os.path.join(path,file)):
        #TODO: issue warning that the model already exists
        pass

    pickle.dump(clf, open(os.path.join(path,file), 'wb'))


def loadClassifier(file, path='models/'):
    if os.path.isfile(os.path.join(path,file)):
        return pickle.load(open(os.path.join(path,file), 'rb'))
    
    return None


if __name__ == '__main__':
    #load the SVM classifier if it exists
    clf = loadClassifier('svm.pickle')

    if clf is None:
        # otherwise train a new SVM classifier
        clf = trainClassifier('samples/RAW/')
        # and save it
        saveClassifier(clf, 'svm.pickle')

    # test the classifier on an image
    image = dataset.getRandomTestImage()
    predictions = clf.predict(image.reshape(-1,3))

    predictions = predictions.reshape(image.shape[:2])
    predictions = cv.normalize(predictions, None, 0, 255, cv.NORM_MINMAX)
    predictions = predictions.astype(np.uint8)
    cv.imshow('predictions',predictions)

    key = cv.waitKey(0)