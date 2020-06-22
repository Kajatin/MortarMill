import logging
import datetime
import json

import pyrealsense2 as rs
import numpy as np
import cv2 as cv

import vision


class Device():
    """ Provides an interface for the D435 device.

    Parameters
    ----------
    config: configparser.SectionProxy
        Configuration file with information about the classifiers to load. If None,
        the segmenter will not function.

    Attributes
    ----------
    serial: string
        Serial number of the device.
    
    Methods
    -------
    startStreaming()
        Starts streaming frames from the device.
    """

    def __init__(self, device, config):
        if type(device) is not rs.pyrealsense2.device:
            raise TypeError(('Argument `device` is expected to be of type: '
                             '<class \'pyrealsense2.device\'>. Instead it was: {}.')
                            .format(type(device)))

        self.logger = logging.getLogger(__name__)

        self.device = device
        self.serial = self.device.get_info(rs.camera_info.serial_number)
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.config = config
        self.filters = []
        self.align = None
        self.depth_scale = None
        self.depth_image = None
        self.color_image = None
        self.nir_left_image = None
        self.nir_right_image = None
        self.logger.info('Setting up device {}.'.format(self.serial))

        # parse the config file if it is not None
        if self.config is None:
            self.logger.error(('No config is provided to Device. Some '
                                 'parameters might not be available, and this '
                                 'module might not work correctly.'))
            raise TypeError('Argument `config` cannot be None.')
        
        # load advanced camera settings onto device
        self.advanced_mode = rs.rs400_advanced_mode(self.device)
        if self.advanced_mode.is_enabled():
            with open(self.config.get('advanced_settings'),'r') as f:
                json_advanced_settings = json.load(f)
            self.advanced_mode.load_json(str(json_advanced_settings).replace("'",'\"'))
            
        # setup post-processing filters from config file
        if self.config.getboolean('use_filters'):
            if self.config.getboolean('use_decimation_filter'):
                self.filters.append(rs.decimation_filter(
                    self.config.getint('decimation')))
            if self.config.getboolean('use_threshold_filter'):
                self.filters.append(rs.threshold_filter(
                    self.config.getfloat('threshold_min'),
                    self.config.getfloat('threshold_max')))
            self.filters.append(rs.disparity_transform(True))
            if self.config.getboolean('use_spatial_filter'):
                self.filters.append(rs.spatial_filter(
                    self.config.getfloat('spatial_alpha'),
                    self.config.getfloat('spatial_delta'),
                    self.config.getfloat('spatial_magnitude'),
                    0))
            if self.config.getboolean('use_temporal_filter'):
                self.filters.append(rs.temporal_filter(
                    self.config.getfloat('temporal_alpha'),
                    self.config.getfloat('temporal_delta'),
                    self.config.getint('temporal_persistence_control')))
            self.filters.append(rs.disparity_transform(False))
            if self.config.getboolean('use_hole_filling'):
                self.filters.append(rs.hole_filling_filter(
                    self.config.getint('hole_filling_mode')))

        # setup depth alignment setting
        if self.config.getboolean('align_depth'):
            self.align = rs.align(rs.stream(2))
            self.logger.info('Depth alignment is enabled on device {}.'.format(self.serial))
        else:
            self.logger.debug('Depth alignment is disabled on device {}.'.format(self.serial))

        # turn depth IR projector on/off
        for sensor in self.device.sensors:
            if sensor.is_depth_sensor():
                self.depth_scale = sensor.as_depth_sensor().get_depth_scale()
                if not self.config.getboolean('assisted_depth'):
                    # turn off the IR emitter
                    sensor.set_option(rs.option.emitter_enabled,0)
                break

        # setting that determins whether undistortion is done or not
        self.do_undistort = self.config.getboolean('undistort')

        # load the camera calibration parameters if they exist
        self.calib_params = vision.calibration.loadCameraCalibrationParams(self.serial)
        if self.calib_params is None:
            self.logger.warning(('Calibration parameters are not available for'
                            ' device {}.').format(self.serial))
        else:
            self.logger.info('Calibration parameters are loaded for device {}'.format(self.serial))
        
        if self.config.get('load_path', None):
            # read from file
            self.rs_config.enable_device_from_file(self.config.get('load_path'))
        else:
            # set up pipeline for a specific device based on S/N
            self.rs_config.enable_device(self.serial)
            # see 'vision_codes_rs.txt' for rs.stream() and rs.format() code clarification
            self.rs_config.enable_stream(rs.stream(1), 848, 480, rs.format(1), 60)    # depth
            self.rs_config.enable_stream(rs.stream(2), 848, 480, rs.format(6), 60)    # colour
            self.rs_config.enable_stream(rs.stream(3), 1, 848, 480, rs.format(9), 60) # left imager
            self.rs_config.enable_stream(rs.stream(3), 2, 848, 480, rs.format(9), 60) # right imager

            if self.config.getboolean('save_to_bag_file'):
                # save to file
                date = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
                save_path = f'samples/recordings/time_{date}_device_{self.serial}.bag'
                self.rs_config.enable_record_to_file(save_path)
        
        # get the built-in intrinsic parameters
        self.startStreaming()
        self.depth_intrinsics = self.profile.get_stream(rs.stream(1))\
                                .as_video_stream_profile()\
                                .get_intrinsics()
        self.colour_intrinsics = self.profile.get_stream(rs.stream(2))\
                                 .as_video_stream_profile()\
                                 .get_intrinsics()
        self.nir_left_intrinsics = self.profile.get_stream(rs.stream(3),1)\
                                   .as_video_stream_profile()\
                                   .get_intrinsics()
        self.nir_right_intrinsics = self.profile.get_stream(rs.stream(3),2)\
                                    .as_video_stream_profile()\
                                    .get_intrinsics()
        self.stopStreaming()


    def __repr__(self):
        return str(self.device)


    def startStreaming(self):
        self.profile = self.pipeline.start(self.rs_config)


    def undistort(self):
        # self calibrated values
        #self.color_image = vision.imgproc.undistortFrame(self.color_image, self.calib_params['colour'])
        #self.nir_left_image = vision.imgproc.undistortFrame(self.nir_left_image, self.calib_params['nir_left'])
        #self.nir_right_image = vision.imgproc.undistortFrame(self.nir_right_image, self.calib_params['nir_right'])
        
        # built-in camera parameters
        self.color_image = vision.imgproc.undistortFrame(self.color_image, self.colour_intrinsics)
        self.nir_left_image = vision.imgproc.undistortFrame(self.nir_left_image, self.nir_left_intrinsics)
        self.nir_right_image = vision.imgproc.undistortFrame(self.nir_right_image, self.nir_right_intrinsics)


    def waitForFrames(self):
        frames = self.pipeline.wait_for_frames()
        # frames.size() -> the number of frames returned

        if self.align is not None:
            frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        nir_left_frame = frames.get_infrared_frame(1)
        nir_right_frame = frames.get_infrared_frame(2)

        # apply filters to depth frame
        for filter in self.filters:
            depth_frame = filter.process(depth_frame)

        # convert images to numpy arrays
        self.color_image = np.asanyarray(color_frame.get_data())
        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.nir_left_image = np.asanyarray(nir_left_frame.get_data())
        self.nir_right_image = np.asanyarray(nir_right_frame.get_data())

        if self.do_undistort:
            self.undistort()

        # scale depth image to meters
        if self.depth_scale is not None:
            self.depth_image = self.depth_image * self.depth_scale

        # print distance at (x,y)
        self.x = int(self.depth_image.shape[0]/2) # row
        self.y = int(self.depth_image.shape[1]/2) # column
        #print(self.depth_image[self.x,self.y])


    def retrieveFrames(self, wait_for_new=True):
        if wait_for_new:
            self.waitForFrames()

        return {'colour':self.color_image,'depth':self.depth_image,
                'nir_left':self.nir_left_image,'nir_right':self.nir_right_image}


    def showFrames(self):
        # apply colormap on depth image (image must be converted to 8-bit per pixel first)
        alpha = 255/np.abs(self.depth_image).max()
        scaled_frame = cv.convertScaleAbs(self.depth_image, alpha=alpha)
        scaled_frame = cv.equalizeHist(scaled_frame)
        depth_colormap = cv.applyColorMap(scaled_frame, cv.COLORMAP_JET)
        print(self.depth_image.shape, depth_colormap.shape)
        
        # draw a marker at pixel where the distance is shown
        #cv.drawMarker(depth_colormap, (self.y,self.x), (0,0,255), cv.MARKER_CROSS,thickness=2)

        cv.imshow(f'RealSense Color {self.serial}', self.color_image)
        cv.imshow(f'RealSense Depth {self.serial}', depth_colormap)
        cv.imshow(f'NIR Left {self.serial}', self.nir_left_image)
        cv.imshow(f'NIR Right {self.serial}', self.nir_right_image)
        

    def stopStreaming(self):
        self.pipeline.stop()