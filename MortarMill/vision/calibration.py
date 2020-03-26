from pathlib import Path
import pickle
import logging
logger = logging.getLogger(__name__)

import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from vision.device import Device


def loadCameraCalibrationParams(serial=None):
    """

    """

    if serial is None:
        logger.warning('`serial` must be specified to load the calibration parameters.')
        return None

    path = Path(f'vision/calibration/device_{serial}/calibration_device_{serial}.pickle')

    if not path.exists():
        logger.info(('Cannot load calibration parameters for device {} '
                     'because they are not available.').format(serial))
        return None

    logger.info('Loading calibration parameters for device {}.'.format(serial))
    return pickle.load(open(path,"rb"))


def calibrateCamera(number_of_ref=14, save=True, monitored=True, force=False):
    """
    works with 9x6 pattern
    https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
    """

    assert isinstance(number_of_ref, int), "Input variable `number_of_ref` must be an integer"


    # make sure that only one device is connected so that the calibration done
    # for the correct device
    devices = rs.context().devices
    if devices.size() != 1:
        logger.error(('To do the calibration, only one device can be connected. '
                      'There were {} device(s) found.').format(devices.size()))
        logger.warning('Calibration is incomplete.')
        return None

    # create a device object from the connected camera
    camera = Device(devices[0],nir=True,assisted_depth=False)

    # create a folder to store the calibration results into
    if save:
        try:
            save_path = f'vision/calibration/device_{camera.serial}'
            Path(save_path).mkdir(parents=True)
            logger.debug('Created folder {} to save the calibration results into.'
                         .format(save_path))
        except FileExistsError:
            if force:
                logger.warning(('The device {} is already calibrated. Since '
                                '`force` is set, the existing calibration will '
                                'be overridden.').format(camera.serial))
                Path(save_path).mkdir(parents=True,exist_ok=True)
            else:
                logger.error(('The device {} is already calibrated. '
                              'Set the `force` parameter to True to overwrite '
                              'existing calibration.').format(camera.serial))

                logger.warning('Calibration is incomplete.')
                return None
    else:
        logger.warning(('The `save` parameter is set to False; the calibration'
                        ' will not be saved.'))

    # start gathering frames the camera (these are retrieved in the while loop below)
    camera.startStreaming()

    # dictionary to store the calibration results into
    calib_matrices = {}

    for key in ['colour','nir_left','nir_right']:
        if monitored:
            logger.info(('Starting camera calibration for device {} stream {} '
                         'in monitored mode.').format(camera.serial,key))
        else:
            logger.info(('Starting camera calibration for device {} stream {} '
                         'in unmonitored mode.').format(camera.serial,key))

        # termination criteria for sub-pixel corner detection
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points array
        objp = np.zeros((9*6,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
            
        # arrays to store object points and image points from all the images
        objpoints = [] # 3D point in real world space
        imgpoints = [] # 2D points in image plane
    
        # start gathering calibration images
        while len(objpoints) < number_of_ref:
            # acquire frames from the camera
            frames = camera.retrieveFrames()

            # select the frame to work with
            frame_calib = frames[key].copy()

            # if no colour image is available continue
            if frame_calib is None:
                continue

            # convert the colour image to grayscale
            if len(frame_calib.shape) == 3:
                gray = cv.cvtColor(frame_calib, cv.COLOR_BGR2GRAY)
            else:
                gray = frame_calib.copy()

            # find the checkerboard corners
            ret, corners = cv.findChessboardCorners(gray, (9,6), None)

            # if found, add object points, image points (after refining them)
            if ret == True:
                # refine corner points at sub-pixel level
                sub_pixel_corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
                if monitored:
                    # draw and display the corners
                    cv.drawChessboardCorners(frame_calib, (9,6), sub_pixel_corners, ret)
                    cv.imshow('Calibration image with pattern', frame_calib)
                
                    ret = cv.waitKey(0)

                    # if `d` is pressed, discard the image and continue
                    if ret == ord('d'):
                        logger.debug('Discarding calibration image (stream: {}) and continuing.'.format(key))
                        continue
                    # if `esc` is pressed, stop the calibration
                    elif ret == 27:
                        camera.stopStreaming()
                        logger.warning('Calibration process is interrupted by user.')
                        logger.warning('Calibration is incomplete.')
                        return None

                # add the found image corners (2D, 3D) to the arrays
                objpoints.append(objp)
                imgpoints.append(sub_pixel_corners)
                logger.debug('Found checkerboard pattern (number {}, stream: {}).'.format(len(objpoints),key))

                # save the calibration image if the `save` argument is set
                if save:
                    save_path = f'vision/calibration/device_{camera.serial}/checkerboard_{len(objpoints)}_{key}.png'
                    if cv.imwrite(save_path,frames[key]):
                        logger.debug('Saving calibration image at {}.'.format(save_path))
                    else:
                        logger.warning('Saving calibration image number {} failed.'.format(len(objpoints)))

        # calculate the camera matrices (rep_err is the RMS reprojection error)
        rep_err, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        logger.info('Camera {} stream {} is calibrated with RMS reprojection error of {}.'.format(camera.serial, key, rep_err))

        # refine the camera matrix
        h, w = frame_calib.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1)

        # add the calibration results into the dictionary
        calib_matrices[key] = {'intrinsics':mtx,'distortion':dist,'new_cam_matrix':newcameramtx,'roi':roi}


    
        # undistort image
        #dst = cv.undistort(frame_calib, mtx, dist, None, newcameramtx)
        dst = cv.undistort(frame_calib, mtx, dist, None)
    
        # crop the image
        x, y, w, h = roi
        #dst = dst[y:y+h, x:x+w]

        cv.imshow('corrected', dst)

        cv.waitKey(0)




    # save the calibration data
    if save:
        save_path = f'vision/calibration/device_{camera.serial}/calibration_device_{camera.serial}.pickle'
        pickle.dump(calib_matrices, open(save_path, 'wb'))
        logger.info('Saving calibration results at {}.'.format(save_path))

    return calib_matrices


def main():
    # calibrate the connected camera
    calib_matrices = calibration.calibrateCamera(save=True)


if __name__ == '__main__':
    main()