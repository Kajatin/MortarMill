import logging

import pyrealsense2 as rs
import numpy as np
import cv2 as cv


class Device():
    """description of class"""

    def __init__(self, device, save=False, load_path=None, nir=False, align=True):
        if device is None:
            pass # TODO: handle case where device is None

        self.logger = logging.getLogger(__name__)

        self.device = device
        self.serial = self.device.get_info(rs.camera_info.serial_number)
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.align = rs.align(rs.stream(2)) if align and not nir else None

        self.logger.info('Setting up device {}'.format(self.serial))

        if load_path:
            # read from file
            self.config.enable_device_from_file(load_path)
        else:
            # set up pipeline for a specific device based on S/N
            self.config.enable_device(self.serial)
            # see 'vision_codes_rs.txt' for rs.stream() and rs.format() code clarification
            self.config.enable_stream(rs.stream(1), 1280, 720, rs.format(1), 30)
            self.config.enable_stream(rs.stream(2), 1920, 1080, rs.format(6), 30)

            if nir:
                # if NIR inputs are needed, the depth stream needs to be disabled
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

        for sensor in self.device.sensors:
            if sensor.is_depth_sensor():
                self.depth_scale = sensor.as_depth_sensor().get_depth_scale()
                break

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
        # frames.size() -> prints the number of frames returned

        if self.align is not None:
            frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        nir_left_frame = frames.get_infrared_frame(1)
        nir_right_frame = frames.get_infrared_frame(2)

        #if not depth_frame or not color_frame:
        #    return None

        # Convert images to numpy arrays
        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())
        #self.nir_left_image = np.asanyarray(nir_left_frame.get_data())
        #self.nir_right_image = np.asanyarray(nir_right_frame.get_data())

        # scale depth image to meters
        self.depth_image = self.depth_image * self.depth_scale

        # print distance at (x,y)
        self.x = int(depth_frame.height/2) # row
        self.y = int(depth_frame.width/2) # column
        print(self.depth_image[self.x,self.y])

        return (self.depth_image, self.color_image)


    def showFrames(self):
        #if self.depth_image == None or self.color_image == None:
        #    return 27

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        alpha = 10#255/np.abs(self.depth_image).max()
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(self.depth_image, alpha=alpha), cv.COLORMAP_JET)

        # draw a marker at pixel where the distance is shown
        cv.drawMarker(depth_colormap, (self.y,self.x), (0,0,255), cv.MARKER_CROSS,thickness=2)

        # Show images
        cv.imshow(f'RealSense Depth {self.serial}', depth_colormap)
        cv.imshow(f'RealSense Color {self.serial}', self.color_image)
        #cv.imshow(f'NIR Left {self.serial}', self.nir_left_image)
        #cv.imshow(f'NIR Right {self.serial}', self.nir_right_image)
        
        return cv.waitKey(1)


    def stopStreaming(self):
        self.pipeline.stop()