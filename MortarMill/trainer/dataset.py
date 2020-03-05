import os

import cv2 as cv
import numpy as np

import common.preprocessing


def loadDatasetPath(basepath):
    dataset = []

    if os.path.isfile(basepath):
        dataset.append(basepath)
    else:
        for entry in os.listdir(basepath):
            path = os.path.join(basepath,entry)
            if os.path.isfile(path):
                dataset.append(path)

                #TODO: fix this function to work with new input structure
                #if 'zoom' in path:
                #    dataset.append(path)

    return dataset


def createTrainingData(dataset, unsupervised=True, ratio=0.05):
    X = []
    y = []

    for path in dataset:
        print(path)
        image = cv.imread(path)
        #image = cv.resize(image, (600, int(image.shape[0] * (600.0/image.shape[1]))))

        CF, TF = common.preprocessing.calculateColorAndTextureFeatures(image.copy())
        features = np.hstack((CF.reshape(-1,3),TF.reshape(-1,1)))

        if unsupervised:
            #TODO: implement case where first all CF,TF features are calculated for all images before the FCM is called
            training = common.preprocessing.assignLabelsUnsupervised(image, features, image.reshape(-1,3), ratio)
    
            X.append(np.vstack([training[0],training[1]]))
            y.append(np.hstack([np.repeat(0,len(training[0])),np.repeat(1,len(training[1]))]))
        else:
            #features_ = features
            features_ = image.reshape(-1,3)

            # select a subset of the data used for training
            selection_features_ = features_[np.random.choice(features_.shape[0], int(features_.shape[0]*ratio), replace=False), :]

            # determine the label from the file name 0: mortar, 1: brick
            label = int(path.split('_')[-1].split('.')[0])

            X.append(selection_features_)
            y.append(np.repeat(label,selection_features_.shape[0]))

    return np.vstack(X), np.hstack(y)


#TODO: code this function properly (getRandomTestImage())
def getRandomTestImage():
    image = cv.imread('samples/RAW/brick_zoom_3.jpg')
    return cv.resize(image, (600, int(image.shape[0] * (600.0/image.shape[1]))))