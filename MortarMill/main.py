import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from path_finder import PathFinder
from device import Device
import trainer.classifier
import trainer.dataset
import common.preprocessing


if __name__ == '__main__':
    # determine if any device is connected or not
    if rs.context().devices.size() > 0:
        camera = Device()
        camera.startStreaming()

        while 1:
            frames = camera.getFrames()
            key = camera.showFrames()

            if key == 27:
                camera.stopStreaming()
                break

    else:
        # load an existing classifier
        clf = trainer.classifier.loadClassifier('svm_test.pickle')

        if clf is None:
            # train a SVM classifier
            clf = trainer.classifier.trainClassifier('samples/train/', False)

            # save it for reuse later
            trainer.classifier.saveClassifier(clf, 'svm_test.pickle')

        # test the classifier on an image trained on RGB
        image = trainer.dataset.getRandomTestImage()
        predictions = clf.predict(image.reshape(-1,3))
        
        # test the classifier on an image trained on CF and TF
        #image = trainer.dataset.getRandomTestImage()
        #CF, TF = common.preprocessing.calculateColorAndTextureFeatures(image.copy())
        #features = np.hstack((CF.reshape(-1,3),TF.reshape(-1,1)))
        #predictions = clf.predict(features)

        predictions = predictions.reshape(image.shape[:2])
        predictions = cv.normalize(predictions, None, 0, 255, cv.NORM_MINMAX)
        predictions = predictions.astype(np.uint8)
        cv.imshow('predictions',predictions)

        cv.imshow('original',image)

        key = cv.waitKey(0)

        exit(0)






        path_finder = PathFinder()
    
        i = 3
        image = cv.imread(f'samples/RAW/brick_zoom_{i}.jpg')
        image = cv.resize(image, (600, int(image.shape[0] * (600.0/image.shape[1]))))

        while(1):
            #key = path_finder.locate_bricks(image, 100, 160)
            #key = path_finder.locate_bricks_hist(image.copy(), 600, 0)
            key = path_finder.locate_bricks_hsv(image.copy())

            if key == 27:
                break