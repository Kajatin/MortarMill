from pprint import pprint
import logging
import datetime

import pyrealsense2 as rs
import cv2 as cv

from device import Device


# TODO: add support for date in log name
#date = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
logging.basicConfig(filename=f'logs/vision.log',
                    level=logging.DEBUG,
                    filemode='w',
                    format='%(asctime)s.%(msecs)05d %(module)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

devices = rs.context().devices
logger.info('Found {} device(s).'.format(devices.size()))

cameras = []
if devices.size() > 0:
    # iterate over devices
    for i, device in enumerate(devices):
        #camera = Device(device, save=True)
        #camera = Device(device, load_path='samples/recordings/time_07032020140142_device_943222071836.bag')
        camera = Device(device)
        
        camera.startStreaming()
        cameras.append(camera)

while 1:
    for camera in cameras:
        frames = camera.getFrames()
        key = camera.showFrames()

    if key == 27:
        for camera in cameras:
            camera.stopStreaming()
        break