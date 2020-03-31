# set up logging
# modules that use logging need to be imported after the logger setup
import logging
import datetime

date = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
logging.basicConfig(filename=f'logs/{date}.log',
                    level=logging.DEBUG,
                    filemode='w',
                    format='%(asctime)s.%(msecs)05d,%(levelname)s,%(pathname)s,%(lineno)d,%(module)s,%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from path_finder import PathFinder
from vision.device import Device


if __name__ == '__main__':
    # determine if any device is connected or not
    devices = rs.context().devices
    logger.info('Found {} device(s).'.format(devices.size()))
    
    path_finder = PathFinder()

    # if no device is connected, use an image input instead for now
    #TODO: hande this part correctly
    if devices.size() <= 0:
        i = 3
        image = cv.imread(f'samples/RAW/brick_zoom_{i}.jpg')
        image = cv.resize(image, (600, int(image.shape[0] * (600.0/image.shape[1]))))

        frames = {'colour':image,'depth':np.zeros(image.shape[:2])}

        path_finder.calibrateHsvThresholds_(image)
        path_finder(frames)

        cv.waitKey(0)
        exit(0)

    cameras = []
    # iterate over devices
    for device in devices:
        #cameras.append(Device(device, align=True))
        cameras.append(Device(device, align=True, load_path='samples/recordings/time_11032020133308_device_943222071836.bag'))

    # only work with one camera for now
    camera = cameras[0]

    # start the stream of frames from the camera
    camera.startStreaming()

    while 1:
        # get the frames from the camera
        frames = camera.retrieveFrames(undistort=False)
        #camera.showFrames()

        # process the frames and show the final mask
        path_finder(frames)

        key = cv.waitKey(1)
        if key == 27:
            camera.stopStreaming()
            break