import os

import cv2 as cv
import numpy as np

import common.preprocessing


def loadDatasetPath(basepath):
    dataset = []
    for entry in os.listdir(basepath):
        path = os.path.join(basepath,entry)
        if os.path.isfile(path):
            if 'zoom' in path: #TODO: fix this function to work with new input structure
                dataset.append(path)

    return dataset


def createTrainingData(dataset, solo=True):
    X = []
    y = []

    for path in dataset:
        print(path)
        image = cv.imread(path)
        image = cv.resize(image, (600, int(image.shape[0] * (600.0/image.shape[1]))))

        CF, TF = common.preprocessing.calculateColorAndTextureFeatures(image.copy())
        features = np.hstack((CF.reshape(-1,3),TF.reshape(-1,1)))

        if solo:
            training = common.preprocessing.assignLabelsUnsupervised(image, features, image.reshape(-1,3))
    
            X.append(np.vstack([training[0],training[1]]))
            y.append(np.hstack([np.repeat(0,len(training[0])),np.repeat(1,len(training[1]))]))

    return np.vstack(X), np.hstack(y)


#TODO: code this function properly (getRandomTestImage())
def getRandomTestImage():
    image = cv.imread('samples/RAW/brick_zoom_3.jpg')
    return cv.resize(image, (600, int(image.shape[0] * (600.0/image.shape[1]))))