import cv2 as cv
import numpy as np

i = 6
image = cv.imread(f'samples/RAW/brick_{i}.jpg')
image = cv.resize(image, (600, int(image.shape[0] * (600.0/image.shape[1]))))

image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#bgr_planes = cv.split(image)
#image = bgr_planes[0]

adaptive_threshold = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 57, 0)

cv.imshow('original', image)
cv.imshow('adaptive_t', adaptive_threshold)
cv.waitKey(0)