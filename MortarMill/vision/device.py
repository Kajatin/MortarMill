import logging
import datetime

import pyrealsense2 as rs
import numpy as np
import cv2 as cv

import vision


class Device():
    """description of class"""

    def __init__(self, device, save=False, load_path=None, nir=False, align=False,
                 assisted_depth=True):
        if device is None:
            pass # TODO: handle case where device is None

        self.logger = logging.getLogger(__name__)

        self.device = device
        self.serial = self.device.get_info(rs.camera_info.serial_number)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.nir = nir

        self.logger.info('Setting up device {}.'.format(self.serial))

        # load the camera calibration parameters if they exist
        self.calib_params = vision.calibration.loadCameraCalibrationParams(self.serial)
        
        if self.calib_params is None:
            self.logger.warning(('Calibration parameters are not available for'
                            ' device {}.').format(self.serial))
        else:
            self.logger.info('Calibration parameters are loaded for device {}'.format(self.serial))
        
        self.align = None
        if align:
            if not self.nir:
                self.align = rs.align(rs.stream(2))
                self.logger.info('Depth alignment is enabled on device {}.'.format(self.serial))
            else:
                self.logger.warning('Tried to enable depth alignment on device {} but the depth stream is disabled in favor of NIR.'.format(self.serial))
        else:
            self.logger.debug('Depth alignment is disabled on device {}.'.format(self.serial))

        if load_path:
            # read from file
            self.config.enable_device_from_file(load_path)
        else:
            # set up pipeline for a specific device based on S/N
            self.config.enable_device(self.serial)
            # see 'vision_codes_rs.txt' for rs.stream() and rs.format() code clarification
            self.config.enable_stream(rs.stream(1), 1280, 720, rs.format(1), 30)
            self.config.enable_stream(rs.stream(2), 1920, 1080, rs.format(6), 30)

            if self.nir:
                # see 'vision_codes_rs.txt' for rs.stream() and rs.format() code clarification
                # if NIR outputs are needed, the depth stream needs to be disabled
                self.config.disable_stream(rs.stream(1))
                self.config.enable_stream(rs.stream(3), 1, 1280, 800, rs.format(9), 30)
                self.config.enable_stream(rs.stream(3), 2, 1280, 800, rs.format(9), 30)

            if save:
                # save to file
                date = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
                save_path = f'samples/recordings/time_{date}_device_{self.serial}.bag'
                self.config.enable_record_to_file(save_path)

        self.depth_image = None
        self.color_image = None
        self.nir_left_image = None
        self.nir_right_image = None

        for sensor in self.device.sensors:
            if sensor.is_depth_sensor():
                self.depth_scale = sensor.as_depth_sensor().get_depth_scale()
                if not assisted_depth:
                    # turn off the IR emitter
                    sensor.set_option(rs.option.emitter_enabled,0)
                break


    def __repr__(self):
        return str(self.device)


    def startStreaming(self):
        self.pipeline.start(self.config)


    def undistort(self):
        if self.calib_params is not None:
            if self.color_image is not None:
                # extract the camera parameters for the colour image
                camera_matrix = self.calib_params['colour']['intrinsics']
                distortion_coeffs = self.calib_params['colour']['distortion']
                new_camera_matrix = self.calib_params['colour']['new_cam_matrix']
                roi = self.calib_params['colour']['roi']

                # built-in calib params
                #for sensor in self.device.sensors:
                #    if sensor.is_color_sensor():
                #        cp = sensor.get_stream_profiles()[0].as_video_stream_profile().get_intrinsics()
                #        camera_matrix = np.array([[cp.fx, 0, cp.ppx],
                #                                 [0, cp.fy, cp.ppy],
                #                                 [0, 0, 1]])
                #        distortion_coeffs = np.array([[0,0,0,0,0]])
                        
                #        break


                # undistort image
                #self.color_image = cv.undistort(self.color_image, camera_matrix,
                #                                distortion_coeffs, None,
                #                                new_camera_matrix)
                self.color_image = cv.undistort(self.color_image, camera_matrix,
                                                distortion_coeffs, None)
    
                # crop the image
                x, y, w, h = roi
                #self.color_image = self.color_image[y:y+h, x:x+w]

            if self.nir_left_image is not None:
                # extract the camera parameters for the left NIR image
                camera_matrix = self.calib_params['nir_left']['intrinsics']
                distortion_coeffs = self.calib_params['nir_left']['distortion']
                new_camera_matrix = self.calib_params['nir_left']['new_cam_matrix']
                roi = self.calib_params['nir_left']['roi']

                # undistort image
                self.nir_left_image = cv.undistort(self.nir_left_image, camera_matrix,
                                                distortion_coeffs, None,
                                                new_camera_matrix)
    
                # crop the image
                x, y, w, h = roi
                #self.nir_left_image = self.nir_left_image[y:y+h, x:x+w]

            if self.nir_right_image is not None:
                # extract the camera parameters for the right NIR image
                camera_matrix = self.calib_params['nir_right']['intrinsics']
                distortion_coeffs = self.calib_params['nir_right']['distortion']
                new_camera_matrix = self.calib_params['nir_right']['new_cam_matrix']
                roi = self.calib_params['nir_right']['roi']

                # undistort image
                self.nir_right_image = cv.undistort(self.nir_right_image, camera_matrix,
                                                distortion_coeffs, None,
                                                new_camera_matrix)
    
                # crop the image
                x, y, w, h = roi
                #self.nir_right_image = self.nir_right_image[y:y+h, x:x+w]


    def waitForFrames(self, undistort=False):
        frames = self.pipeline.wait_for_frames()
        # frames.size() -> prints the number of frames returned

        if self.align is not None:
            frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        nir_left_frame = frames.get_infrared_frame(1)
        nir_right_frame = frames.get_infrared_frame(2)

        # convert images to numpy arrays
        self.color_image = np.asanyarray(color_frame.get_data())
        if not self.nir:
            self.depth_image = np.asanyarray(depth_frame.get_data())

            # scale depth image to meters
            self.depth_image = self.depth_image * self.depth_scale

            # print distance at (x,y)
            self.x = int(depth_frame.height/2) # row
            self.y = int(depth_frame.width/2) # column
            print(self.depth_image[self.x,self.y])
        else:
            self.nir_left_image = np.asanyarray(nir_left_frame.get_data())
            self.nir_right_image = np.asanyarray(nir_right_frame.get_data())

        if undistort:
            self.undistort()


    def retrieveFrames(self, wait_for_new=True, undistort=False):
        if wait_for_new:
            self.waitForFrames(undistort)

        return {'colour':self.color_image,'depth':self.depth_image,
                'nir_left':self.nir_left_image,'nir_right':self.nir_right_image}


    def showFrames(self):
        if not self.nir:
            # apply colormap on depth image (image must be converted to 8-bit per pixel first)
            alpha = 10#255/np.abs(self.depth_image).max()
            depth_colormap = cv.applyColorMap(cv.convertScaleAbs(self.depth_image, alpha=alpha), cv.COLORMAP_JET)

            # draw a marker at pixel where the distance is shown
            cv.drawMarker(depth_colormap, (self.y,self.x), (0,0,255), cv.MARKER_CROSS,thickness=2)

            cv.imshow(f'RealSense Depth {self.serial}', depth_colormap)
        else:
            cv.imshow(f'NIR Left {self.serial}', self.nir_left_image)
            cv.imshow(f'NIR Right {self.serial}', self.nir_right_image)
            
        cv.imshow(f'RealSense Color {self.serial}', self.color_image)
        

    def stopStreaming(self):
        self.pipeline.stop()