import cv2 as cv
import numpy as np

i = 6
image = cv.imread(f'samples/RAW/brick_{i}.jpg',0)
image = cv.resize(image, (600, int(image.shape[0] * (600.0/image.shape[1]))))

f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

magnitude_spectrum_out = cv.normalize(magnitude_spectrum, None, 0, 255, cv.NORM_MINMAX)
magnitude_spectrum_out = magnitude_spectrum_out.astype(np.uint8)

cv.imshow('input',image)
cv.imshow('magnitude_spectrum',magnitude_spectrum_out)



rows, cols = image.shape
crow,ccol = rows//2 , cols//2
fshift[crow-55:crow+56, ccol-55:ccol+56] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)

img_back_out = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX)
img_back_out = img_back_out.astype(np.uint8)

cv.imshow('filtered',img_back)




cv.waitKey(0)