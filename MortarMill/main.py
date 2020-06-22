import logging
import datetime
import os

from confighandler import ConfigHandler
ch = ConfigHandler('config.ini')

# set up logging
# modules that use logging need to be imported after the logger setup
# create log folder if it does not exist
if not os.path.isdir('logs/'):
    os.makedirs('logs/')
date = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
logging.basicConfig(level=ch.getLogLevel(),
                    format='%(asctime)s.%(msecs)05d,%(levelname)s,%(pathname)s,%(lineno)d,%(module)s,%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler(f'logs/{date}.log'),
                        logging.StreamHandler()
                        ])
logger = logging.getLogger(__name__)

import cv2 as cv
import numpy as np
import pyrealsense2 as rs

import vision
import ml
from segmenter import Segmenter


if __name__ == '__main__':
    # determine if any device is connected or not
    devices = rs.context().devices
    logger.info('Found {} device(s).'.format(devices.size()))
    
    segmenter = Segmenter(ch.config['SEGMENTER'])

    # if no device is connected, use an image input instead for now
    if devices.size() <= 0:
        i=10
        image = cv.imread(f'samples/eval/colour_{i}.png')
        depth = np.load(f'samples/eval/depth_{i}.npy')
        frames = {'colour':image,'depth':depth}

        ret = vision.imgproc.calibrateHsvThresholds(frames['colour'],False)
        segmenter.lowerb, segmenter.upperb = ret

        mask = segmenter(frames)

        vision.imgproc.visualizeFinalSegmentation(frames['colour'],mask)
        cv.waitKey(0)
        exit(0)

    cameras = []
    # iterate over devices
    for device in devices:
        cameras.append(vision.Device(device, ch.config['DEVICE']))

    # only work with one camera for now
    camera = cameras[0]

    # start the stream of frames from the camera
    camera.startStreaming()

    while 1:
        # get the frames from the camera
        frames = camera.retrieveFrames()
        camera.showFrames()

        # process the frames and show the final mask
        #mask = segmenter(frames)

        #vision.imgproc.visualizeFinalSegmentation(frames['colour'],mask)
        key = cv.waitKey(1)
        if key == 27:
            camera.stopStreaming()
            break