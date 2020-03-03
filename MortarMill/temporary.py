#import cv2 as cv
#import numpy as np

#hue_min = 0
#hue_max = 180

#def hue_min_callback(val):
#    global hue_min
#    global hue_max
#    hue_min = val

#    # Convert BGR to HSV
#    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
#    # define range of blue color in HSV
#    lower_blue = np.array([hue_min,50,50])
#    upper_blue = np.array([hue_max,255,255])
#    # Threshold the HSV image to get only blue colors
#    mask = cv.inRange(hsv, lower_blue, upper_blue)
#    # Bitwise-AND mask and original image
#    res = cv.bitwise_and(image,image, mask= mask)

#    cv.imshow('res',res)

#def hue_max_callback(val):
#    global hue_min
#    global hue_max
#    hue_max = val

#    # Convert BGR to HSV
#    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
#    # define range of blue color in HSV
#    lower_blue = np.array([hue_min,50,50])
#    upper_blue = np.array([hue_max,255,255])
#    # Threshold the HSV image to get only blue colors
#    mask = cv.inRange(hsv, lower_blue, upper_blue)
#    # Bitwise-AND mask and original image
#    res = cv.bitwise_and(image,image, mask= mask)

#    cv.imshow('res',res)

#cv.namedWindow('res')
#cv.createTrackbar('hue_min','res',0,180,hue_min_callback)
#cv.createTrackbar('hue_max','res',0,180,hue_max_callback)

#i = 6
#image = cv.imread(f'samples/RAW/brick_{i}.jpg')

#if image is not None:
#    print('before: {}'.format(image.shape))
#    image = cv.resize(image, (600, int(image.shape[0] * (600.0/image.shape[1]))))
#    print('after: {}'.format(image.shape))
    
    
#    #cv.imshow('image',image)
#    #cv.imshow('mask',mask)
#    #cv.imshow('res',res)
    
#    key = cv.waitKey(0)
#    #if key == 27:
#    #    break

#cv.destroyAllWindows()






#import numpy as np
#import cv2

## cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
## ksize - size of gabor filter (n, n)
## sigma - standard deviation of the gaussian function
## theta - orientation of the normal to the parallel stripes
## lambda - wavelength of the sunusoidal factor
## gamma - spatial aspect ratio
## psi - phase offset
## ktype - type and range of values that each pixel in the gabor kernel can hold

#g_kernel = cv2.getGaborKernel((21, 21), 8.0, 0, 10.0, 0.5, 0, ktype=cv2.CV_32F)

#i = 6
#image = cv2.imread(f'samples/RAW/brick_{i}.jpg')
#img = cv2.resize(image, (600, int(image.shape[0] * (600.0/image.shape[1]))))
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

#cv2.imshow('image', img)
#cv2.imshow('filtered image', filtered_img)

#h, w = g_kernel.shape[:2]
#g_kernel = cv2.resize(g_kernel, (30*w, 30*h), interpolation=cv2.INTER_CUBIC)
#cv2.imshow('gabor kernel (resized)', g_kernel)
#cv2.waitKey(0)
#cv2.destroyAllWindows()











import pyrealsense2 as rs
import numpy as np
import cv2

points = rs.points()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
profile = pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        nir_lf_frame = frames.get_infrared_frame(1)
        nir_rg_frame = frames.get_infrared_frame(2)
        if not nir_lf_frame or not nir_rg_frame:
            continue
        nir_lf_image = np.asanyarray(nir_lf_frame.get_data())
        nir_rg_image = np.asanyarray(nir_rg_frame.get_data())
        # horizontal stack
        image=np.hstack((nir_lf_image,nir_rg_image))
        cv2.namedWindow('NIR images (left, right)', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('IR Example', image)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()






#import pyrealsense2 as rs2
##import numpy as np
#import cv2 as cv
#from pprint import pprint

#devices = rs2.context().devices
#sensors = rs2.context().sensors # all sensors

#print('Found {} device(s).\n'.format(devices.size()))

## get the first device
#if devices.size() > 0:
#    device1 = devices.front()
    
#    print('Sensors found for device {} ({}):'.format(device1.get_info(rs2.camera_info.name),device1.get_info(rs2.camera_info.serial_number)))
#    device1_sensors = device1.sensors
#    for i, sensor in enumerate(device1_sensors):
#        print('{}: {}'.format(i,sensor.get_info(rs2.camera_info.name)))
#    print()

#    print('Available sensor profiles for {}:'.format(device1_sensors[0].get_info(rs2.camera_info.name)))
#    pprint(device1_sensors[0].profiles)
#    print()
#    print('Available sensor profiles for {}:'.format(device1_sensors[1].get_info(rs2.camera_info.name)))
#    pprint(device1_sensors[0].profiles)



cv.namedWindow('dummy')
cv.waitKey(0)