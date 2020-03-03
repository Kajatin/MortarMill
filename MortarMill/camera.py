import pyrealsense2 as rs
import numpy as np
import cv2 as cv

class Camera():
    """description of class"""

    def __init__(self, config=None):
        self.pipeline = rs.pipeline()

        if config:
            self.config = config
        else:
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            #self.config.enable_all_streams()

        self.depth_image = None
        self.color_image = None


    def startStreaming(self, with_config=True):
        if with_config:
            self.pipeline.start(self.config)
        else:
            self.pipeline.start()

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


        # Stack both images horizontally
        images = np.hstack((self.color_image, depth_colormap))


        # Show images
        cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
        cv.imshow('RealSense', images)
        
        return cv.waitKey(1)

    def stopStreaming(self):
        self.pipeline.stop()