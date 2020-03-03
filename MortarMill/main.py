import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from path_finder import PathFinder
from camera import Camera


if __name__ == '__main__':
    # determine if any device is connected or not
    if rs.context().devices.size() > 0:
        camera = Camera()
        camera.startStreaming()

        while 1:
            frames = camera.getFrames()
            key = camera.showFrames()

            if key == 27:
                camera.stopStreaming()
                break

    else:
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