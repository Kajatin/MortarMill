from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np

i = 6
image_original = cv.imread(f'samples/RAW/brick_{i}.jpg')
image_o = cv.resize(image_original, (600, int(image_original.shape[0] * (600.0/image_original.shape[1]))))
image = image_o / 255.0

image_n = image.reshape(image.shape[0]*image.shape[1], image.shape[2])

kmeans = KMeans(n_clusters = 2).fit(image_n)

pic2show = kmeans.cluster_centers_[kmeans.labels_]
cluster_pic = pic2show.reshape(image.shape[0], image.shape[1], image.shape[2])

cv.imshow('original',image_o)
cv.imshow('cluster',cluster_pic)
cv.waitKey(0)