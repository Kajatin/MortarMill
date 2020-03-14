import pickle
import logging
logger = logging.getLogger(__name__)

import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from vision.device import Device


def calibrateCamera(number_of_ref=14, save=True):
    """
    https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
    """

    assert isinstance(number_of_ref, int), "Input variable `number_of_ref` must be an integer"

    devices = rs.context().devices
    if devices.size() != 1:
        logger.error('To do the calibration, only one device can be connected. There were {} device(s) found.'.format(devices.size()))
        return False, None

    camera = Device(devices[0])
    camera.startStreaming()
    logger.info('Starting camera calibration for device {}.'.format(camera.serial))

    # termination criteria for sub-pixel corner detection
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points array
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
            
    # arrays to store object points and image points from all the images
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane
    
    while len(objpoints) < number_of_ref:
        # acquire frames from the camera
        frames = camera.getFrames()
        #camera.showFrames()

        # extract the colour image output
        img = frames[1].copy()
        # convert the colour image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (9,6), None)

        # if found, add object points, image points (after refining them)
        if ret == True:
            # refine corner points at sub-pixel level
            sub_pixel_corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(sub_pixel_corners)
            
            # draw and display the corners
            cv.drawChessboardCorners(img, (9,6), sub_pixel_corners, ret)
            cv.imshow('img', img)

            logger.debug('Found checkerboard pattern (number {}).'.format(len(objpoints)))

            if cv.waitKey(0) == 27:
                camera.stopStreaming()
                break

    # calculate the camera matrices (rep_err is the RMS reprojection error)
    rep_err, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    logger.info('Camera {} is calibrated with RMS reprojection error of {}.'.format(camera.serial, rep_err))

    # refine the camera matrix
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # group the calibration results into an array
    calib_matrices = [mtx, dist, newcameramtx]

    # save the calibration data
    if save:
        save_path = f'vision/calibration/calibration_device_{camera.serial}.pickle'
        pickle.dump(calib_matrices, open(save_path, 'wb'))
        logger.info('Saving calibration results at {}.'.format(save_path))





    # undistort image
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    
    # crop the image
    x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]

    cv.imshow('corrected', dst)

    cv.waitKey(0)

    return True, calib_matrices

if __name__ == '__main__':
    main()

    mtx, dist, newcameramtx = pickle.load(open("calibration.pickle","rb"))

    # undistort image
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    cv.imshow('corrected_', dst)

    cv.waitKey(0)