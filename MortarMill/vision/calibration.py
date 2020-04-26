from pathlib import Path
import pickle
import logging
logger = logging.getLogger(__name__)

from confighandler import ConfigHandler
ch = ConfigHandler('config.ini')

import cv2 as cv
import numpy as np
import pyrealsense2 as rs

import vision


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


def calibrateCamera(camera, save=True, force=False):
    # create a folder to store the calibration results into if `save` is True
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
                return False
    else:
        logger.warning(('The `save` parameter is set to False; the calibration'
                        ' will not be saved.'))

    intrinsic_matrices = calibrateCameraIntrinsics(camera, save=save)
    ret, rvec, tvec = calibrateCameraExtrinsics(camera, intrinsic_matrices)
    #ret, rvec, tvec = calibrateCameraExtrinsics(camera, camera.colour_intrinsics)

    if ret:
        calib_matrices = intrinsic_matrices
        calib_matrices['colour']['extrinsics'] = (rvec, tvec)

        # save the calibration data
        if save:
            save_path = f'vision/calibration/device_{camera.serial}/calibration_device_{camera.serial}.pickle'
            pickle.dump(calib_matrices, open(save_path, 'wb'))
            logger.info('Saving calibration results in {}.'.format(save_path))
        return True
    else:
        logger.warning('Calibration of device {} is unsuccessful.'.format(camera.serial))
        return False


def calibrateCameraIntrinsics(camera, number_of_ref=14, save=True, monitored=True):
    """
    works with 9x6 pattern
    https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
    """

    assert isinstance(number_of_ref, int), "Input variable `number_of_ref` must be an integer"

    # start gathering frames from the camera (these are retrieved in the while loop below)
    camera.startStreaming()

    # dictionary to store the calibration results into
    calib_matrices = {}

    # start calibration for each of the streams
    for key in ['colour','nir_left','nir_right']:
        if monitored:
            logger.info(('Starting camera intrinsic calibration for device {} stream {} '
                         'in monitored mode.').format(camera.serial,key))
        else:
            logger.info(('Starting camera intrinsic calibration for device {} stream {} '
                         'in unmonitored mode.').format(camera.serial,key))

        # termination criteria for sub-pixel corner detection
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points array (3D world)
        objp = np.zeros((9*6,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        objp *= 24
            
        # arrays to store object points and image points from all the images
        objpoints = [] # 3D points in real world space
        imgpoints = [] # 2D points in image plane
    
        # start gathering calibration images
        while len(objpoints) < number_of_ref:
            # acquire frames from the camera
            frames = camera.retrieveFrames()

            # select the frame to work with
            frame_calib = frames[key].copy()

            # if no colour image is available continue
            if frame_calib is None:
                logger.debug(('Skipping current frame during intrinsic calibration'
                             ' since it is None.'))
                continue

            # convert the colour image to grayscale
            if len(frame_calib.shape) == 3:
                gray = cv.cvtColor(frame_calib, cv.COLOR_BGR2GRAY)
            else:
                gray = frame_calib.copy()

            # find the checkerboard corners
            ret, corners = cv.findChessboardCorners(gray, (9,6), None)

            # if found, add object points, image points (after refining them)
            if ret:
                # refine corner points at sub-pixel level
                sub_pixel_corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
                if monitored:
                    # draw and display the corners
                    cv.drawChessboardCorners(frame_calib, (9,6), sub_pixel_corners, ret)
                    cv.imshow('Intrinsic calibration', frame_calib)
                
                    ret = cv.waitKey(0)
                    # if `d` is pressed, discard the image and continue
                    if ret == ord('d'):
                        logger.debug('Discarding calibration image (stream: {}) and continuing.'.format(key))
                        continue
                    # if `esc` is pressed, stop the calibration
                    elif ret == 27:
                        camera.stopStreaming()
                        logger.warning(('Intrinsic calibration process is interrupted'
                                        ' by user. Calibration is incomplete.'))
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
        logger.info(('Camera {} stream {} intrinsic parameters are calibrated '
                    'with RMS reprojection error of {}.').format(camera.serial, key, rep_err))

        # refine the camera matrix
        h, w = frame_calib.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1)

        # add the calibration results into the dictionary
        calib_matrices[key] = {'intrinsics':mtx,
                               'distortion':dist,
                               'new_cam_matrix':newcameramtx,
                               'roi':roi}

    # stop streaming from the camera
    camera.stopStreaming()

    return calib_matrices


def calibrateCameraExtrinsics(camera, params, stream='colour'):
    if params is None:
        logger.warning(('Cannot calibrate extrinsic parameters because the '
                       '`params` argument is None.'))
        return False, None, None

    # start gathering frames from the camera (these are retrieved in the while loop below)
    camera.startStreaming()

    logger.info('Starting camera extrinsic calibration for device {} stream {}.'\
        .format(camera.serial,stream))

    # return variables
    success = False
    rvec = None
    tvec = None

    # load camera calibration parameters
    if isinstance(params, dict):
        intrinsics = params[stream]['intrinsics']
        distortion = params[stream]['distortion']
    else:
        intrinsics = np.array([[params.fx, 0, params.ppx],
                               [0, params.fy, params.ppy],
                               [0, 0, 1]])
        distortion = np.array([params.coeffs])

    # termination criteria for sub-pixel corner detection
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points array
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objp *= 24

    while 1:
        # acquire frames from the camera
        frames = camera.retrieveFrames()

        # select the frame to work with
        frame_calib = frames[stream].copy()

        # if no colour image is available continue
        if frame_calib is None:
            return

        # convert the colour image to grayscale
        if len(frame_calib.shape) == 3:
            gray = cv.cvtColor(frame_calib, cv.COLOR_BGR2GRAY)
        else:
            gray = frame_calib.copy()

        # find the checkerboard corners
        ret, corners = cv.findChessboardCorners(gray, (9,6), None)

        # if found, add object points, image points (after refining them)
        if ret:
            # refine corner points at sub-pixel level
            sub_pixel_corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            # draw and display the detected corners
            cv.drawChessboardCorners(frame_calib, (9,6), sub_pixel_corners, ret)
            # draw a marker at camera centre
            ppx = int(intrinsics[0,2])
            ppy = int(intrinsics[1,2])
            cv.drawMarker(frame_calib,(ppx, ppy),(0,0,255),cv.MARKER_CROSS,thickness=2)
            cv.imshow('Extrinsic calibration', frame_calib)

            key = cv.waitKey(1)
            if key == 27:
                camera.stopStreaming()
                logger.warning(('Extrinsic calibration process is interrupted'
                                ' by user. Calibration is incomplete.'))
                break
            elif key == ord('c'):
                # estimate the extrinsic parameters
                success, rvec, tvec = cv.solvePnP(objp, sub_pixel_corners, intrinsics, distortion)
                
                if success:
                    logger.info(('Camera {} stream {} extrinsic parameters are '
                                 'calibrated. rvec: {} tvec: {}')\
                                     .format(camera.serial,stream,rvec,tvec))
                else:
                    logger.debug(('Camera {} stream {} extrinsic parameters are '
                                 'not calibrated.').format(camera.serial,stream))
                break

    camera.stopStreaming()
    return success, rvec, tvec


def pointsToWorld(frames, intrinsics, distortion, rvec, tvec):
    """ Converts image points to world coordinates """

    # height and width of the depth image
    h, w, *_ = frames['depth'].shape
    # generate grid of points to be transformed to world coordinates
    points = np.insert(np.mgrid[0:h,0:w].reshape(2,-1),2,1,axis=0)
    # 3x3 rotation matrix from the 3 Euler angles
    rotm, _ = cv.Rodrigues(rvec)

    # first determine the arbitrary scale factor `s` based on the depth
    leftSideMat  = np.linalg.inv(rotm) @ np.linalg.inv(intrinsics) @ points
    rightSideMat = np.linalg.inv(rotm) @ tvec
    Z = frames['depth']*1000 - tvec[2,0]
    s = (Z.ravel() + rightSideMat[2,0]) / leftSideMat[2,:]

    # using `s`, transform the image points to world coordinates
    world = np.linalg.inv(rotm) @ (s * (np.linalg.inv(intrinsics) @ points) - tvec)
    world = world.T.reshape(h,w,3)

    # set elements to 0 where the depth value was also 0 (invalid)
    i,j = np.asarray(frames['depth']==0).nonzero()
    world[i,j,:] = 0

    return world


def main():
    devices = rs.context().devices
    # create a Device instance for the camera to be calibrated
    camera = vision.Device(devices[0], ch.config['DEVICE'])

    # calibrate the connected camera
    calibrateCamera(camera)

if __name__ == '__main__':
    main()