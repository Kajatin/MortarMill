import datetime
import logging

import pyrealsense2 as rs
import numpy as np
import cv2 as cv

logging.basicConfig(filename=f'logs/vision.log',
                    level=logging.DEBUG,
                    filemode='w',
                    format='%(asctime)s.%(msecs)05d %(module)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class Device():
    """description of class"""

    def __init__(self, device, save=False, load_path=None):

        self.device = device
        self.serial = self.device.get_info(rs.camera_info.serial_number)
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if load_path:
            # read from file
            self.config.enable_device_from_file(load_path)
        else:
            # set up pipeline for a specific device based on S/N
            self.config.enable_device(self.serial)
            # see 'vision_codes_rs.txt' for rs.stream() and rs.format() code clarification
            self.config.enable_stream(rs.stream(1), 1280, 720, rs.format(1), 30)
            self.config.enable_stream(rs.stream(2), 1920, 1080, rs.format(6), 30)

            if save:
                # save to file
                date = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
                save_path = f'samples/recordings/time_{date}_device_{self.serial}.bag'
                self.config.enable_record_to_file(save_path)

        self.depth_image = None
        self.color_image = None

        # TODO: add logging
        #print('Sensors found for device {} ({}):'.format(device.get_info(rs.camera_info.name),device.get_info(rs.camera_info.serial_number)))
        #for i, sensor in enumerate(device.sensors):
        #    print('{}: {}'.format(i,sensor.get_info(rs.camera_info.name)))
        #print()


    def __repr__(self):
        return str(self.device)


    def startStreaming(self):
        self.pipeline.start(self.config)


    def getFrames(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        # print distance at (x,y)
        self.x = 240
        self.y = 320
        dist = depth_frame.get_distance(self.x,self.y)
        print(dist)

        # Convert images to numpy arrays
        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())

        return (self.depth_image, self.color_image)


    def showFrames(self):
        #if self.depth_image == None or self.color_image == None:
        #    return 27

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(self.depth_image, alpha=0.03), cv.COLORMAP_JET)

        cv.drawMarker(self.color_image, (self.x,self.y), (0,0,255), cv.MARKER_CROSS,thickness=2)
        cv.drawMarker(depth_colormap, (self.x,self.y), (0,0,255), cv.MARKER_CROSS,thickness=2)

        # Show images
        cv.imshow('RealSense Depth', depth_colormap)
        cv.imshow('RealSense Color', self.color_image)
        
        return cv.waitKey(1)


    def stopStreaming(self):
        self.pipeline.stop()