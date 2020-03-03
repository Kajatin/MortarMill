import cv2 as cv
import numpy as np
import math
from scipy.stats import kurtosis

from sklearn.cluster import KMeans

from FCM import FCM


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PathFinder(metaclass=Singleton):
    """description of class"""

    def __init__(self):
        pass


    def locate_edges(self, rows, threshold):
        edges = []

        prev = -1
        for i,row in enumerate(rows):
            if row - prev > threshold:
                edges.append(rows[i-1])
                edges.append(row)
            prev = row

        return edges


    def locate_bricks_hsv(self, frame):
        low_H = 0
        low_S = 35
        low_V = 180
        high_H = 180
        high_S = 255
        high_V = 255

        # copy of original input
        frame_original = frame.copy()
        
        # hsv thresholding
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        closing = cv.morphologyEx(frame_threshold, cv.MORPH_CLOSE, kernel)
        
        # connected components
        nb_comp, label, stats, centroids = cv.connectedComponentsWithStats(closing, connectivity=8)
        min_size = 250

        cc_out = np.zeros((closing.shape))
        for i in range(1,nb_comp):
            if stats[i, -1] >= min_size:
                cc_out[label == i] = 255

        # connected components inverted
        inverted_image = np.zeros((cc_out.shape),np.uint8)
        inverted_image[cc_out == 0] = 255

        nb_comp_i, label_i, stats_i, centroids_i = cv.connectedComponentsWithStats(inverted_image, connectivity=8)
        min_size = 2000

        cc_out_inverted = np.zeros((closing.shape))
        for i in range(1,nb_comp_i):
            if stats_i[i,-1] >= min_size:
                cc_out_inverted[label_i == i] = 255

        # final mask
        final_mask = np.zeros((cc_out_inverted.shape),np.uint8)
        final_mask[cc_out_inverted == 0] = 255
        #cv.imshow('final mask', final_mask)
    
        mask = final_mask
        res = cv.bitwise_and(frame,frame,mask=mask)
    
        # detect bricks
        nb_comp_f, label_f, stats_f, centroids_f = cv.connectedComponentsWithStats(mask, connectivity=8)
        for center in centroids_f[1:]:
            cv.drawMarker(frame, tuple(np.uint(center)), (255,0,0), cv.MARKER_CROSS,thickness=2)

        for stat in np.uint(stats_f[1:]):
            cv.rectangle(frame, (stat[0],stat[1]),(stat[0]+stat[2],stat[1]+stat[3]),(0,0,255),2)

        # detect contours
        cnts, hrcy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(frame, cnts, -1, (0,255,0),2)
    
        #cv.imshow('input',frame)
        #cv.imshow('bricks',res)
        cv.imshow('output',np.vstack((frame_original,res,frame)))
        #cv.imwrite('output.png',np.hstack((res,frame)))

        return cv.waitKey(0)


    def locate_bricks_hist(self, image, col_width, limit):
        if col_width < 10:
            col_width = 10

        image_row = image.copy()
        image_col = image.copy()

        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

        # horizontal
        start = 0
        end = col_width
        step = 1
        for j in range(gray.shape[1]//col_width):
            rows = []
            for i in range(0,gray.shape[0],step):
                kurt = kurtosis(gray[i:i+step,start:end],None)
            
                if (kurt <= limit):
                    rows.append(i)
                    cv.line(image_col,(start,i),(end,i),(0,255,0),1)
                    #cv.line(image_row,(start,i),(end,i),(0,255,0),1)

            edges = self.locate_edges(rows,10)
            for edge in edges:
                cv.line(image_row,(start,edge),(end,edge),(0,0,255),2)

            start = end
            end = end + col_width


        # vertical
        start = 0
        end = col_width
        step = 10
        for j in range(gray.shape[0]//col_width):
            for i in range(0,gray.shape[1],step):
                kurt = kurtosis(gray[start:end,i:i+step],None)
            
                #if (kurt <= limit):
                    #cv.line(image_col,(i,start),(i,end),(0,255,0),1)

            start = end
            end = end + col_width
        
        cv.imshow('original',image)
        cv.imshow('im_row',image_row)    
        cv.imshow('im_col',image_col)
    
        return cv.waitKey(0)


    def calculateColorAndTextureFeatures(self, image):
        ## Step 1: Pixel-level COLOR features extraction
        # convert color space to CIE L*a*b
        image_lab = cv.cvtColor(image,cv.COLOR_BGR2Lab)
        #image_lab = image
        #TODO: check what color space to use

        # convert dtype to float to avoid overflow during computations
        image_lab_f = image_lab.astype(float)

        # standard deviation of color component within window over image
        # sigma = sqrt(E[X^2] - E[X]^2)
        kernel_size = (5,5)
        mu = cv.blur(image_lab_f, kernel_size) # E[X]
        mu2 = cv.blur(image_lab_f**2, kernel_size) # E[X^2]
        v = cv.sqrt(mu2 - mu**2) # std of each color component

        # normalize sigma values between 0-1 for each channel
        normed_v = cv.normalize(v, None, 0, 1, cv.NORM_MINMAX)
        
        # sobel
        kernel_size = 5
        dx = cv.Sobel(image_lab_f, -1, 1, 0, ksize=kernel_size)
        dy = cv.Sobel(image_lab_f, -1, 0, 1, ksize=kernel_size)
        e = cv.sqrt(dx**2 + dy**2) # local discontinuity

        # normalize e values between 0-1 for each channel
        normed_e = cv.normalize(e, None, 0, 1, cv.NORM_MINMAX)

        # pixel level color feature (local homogeneity)
        # the more uniform a local region is, the larger the homogeneity value is
        CF = 1 - normed_e * normed_v
        #CF = 1 - np.cross(normed_e,normed_v)
        #TODO: check which version is ok

        ## Step 2: Pixel-level TEXTURE features extraction
        # convert color space to YCrCb
        image_ycrcb = cv.cvtColor(image,cv.COLOR_BGR2YCrCb)

        # convert dtype to float to avoid overflow during computations
        image_ycrcb_f = image_ycrcb.astype(float)
        image_y = image_ycrcb_f[:,:,0].copy()

        # create and apply Gabor filter to luminance component of image
        filters = self.generateGaborFilters(
            np.linspace(7,8,3),
            #np.linspace(1,np.pi,10),
            np.linspace(0,np.pi/2,2),
            #np.linspace(0,np.pi,4),
            #(15,15),
            None,
            np.pi*0.75)

        # apply Gabor filter
        filtered_images = np.zeros((image_y.shape[0],image_y.shape[1],len(filters)))
        for i, filter in enumerate(filters):
            filtered_images[:,:,i] = cv.filter2D(image_y, -1, filter[0])

        # pixel texture feature extraction
        filtered_images = cv.normalize(filtered_images, None, -1, 1, cv.NORM_MINMAX)
        TF = np.abs(filtered_images).max(2)


        ###################
        # will be removed
        final_color = cv.normalize(CF, None, 0, 255, cv.NORM_MINMAX)
        final_color = final_color.astype(np.uint8)
        #final_color = cv.cvtColor(final_color, cv.COLOR_Lab2BGR)
        cv.imshow('final_color',final_color)

        final_texture = cv.normalize(TF, None, 0, 255, cv.NORM_MINMAX)
        final_texture = final_texture.astype(np.uint8)
        cv.imshow('final_texture',final_texture)

        # visualize filters
        for i,filter in enumerate(filters):
            filter_to_show = cv.resize(filter[0],(200,200))
            filter_to_show = cv.normalize(filter_to_show, None, 0, 255, cv.NORM_MINMAX)
            filter_to_show = filter_to_show.astype(np.uint8)
            cv.imshow(f'filter_{i}',filter_to_show)

        # visualize filtered images
        #for i in range(filtered_images.shape[2]):
        #    filter_to_show = cv.normalize(filtered_images[:,:,i], None, 0, 255, cv.NORM_MINMAX)
        #    filter_to_show = filter_to_show.astype(np.uint8)
        #    cv.imshow(f'filtered_image_{i}',filter_to_show)





        ##################
        # fuzzy c-means
        # fit the fuzzy-c-means
        fcm = FCM(n_clusters=2,m=1.3,max_iter=1500,error=1e-7,random_state=np.random.randint(10000))
        fcm.fit(np.hstack((CF.reshape(-1,3),TF.reshape(-1,1))))
        #fcm.fit(TF.reshape(-1,1))
        #fcm.fit(CF.reshape(-1,3))

        # outputs
        fcm_centers = fcm.centers
        fcm_labels  = fcm.u.argmax(axis=1)
        unique, counts = np.unique(fcm_labels, return_counts=True)
        print(fcm_centers)
        print(unique, counts)

        ratio = 0.05
        fcm_labels = np.ones((image.shape[0]*image.shape[1],)) * 0.5
        for i in unique:
            args = fcm.u.argpartition(-int(counts[i]*ratio),axis=0)[-int(counts[i]*ratio):]
            fcm_labels[args[:,i]] = i

        fcm_labels = fcm_labels.reshape(image.shape[:2])
        fcm_labels = cv.normalize(fcm_labels, None, 0, 255, cv.NORM_MINMAX)
        fcm_labels = fcm_labels.astype(np.uint8)
        cv.imshow('fcm_labels',fcm_labels)

        fcm_labels  = fcm.u.argmax(axis=1)
        fcm_labels = fcm_labels.reshape(image.shape[:2])
        fcm_labels = cv.normalize(fcm_labels, None, 0, 255, cv.NORM_MINMAX)
        fcm_labels = fcm_labels.astype(np.uint8)
        cv.imshow('fcm_labels_orig',fcm_labels)

        cv.imshow('original',image)

        return cv.waitKey(0)


    def locate_bricks(self, image_arg, min, max):
        low_H = 0
        low_S = 20
        low_V = 180
        high_H = 20
        high_S = 255
        high_V = 255

        frame = image_arg.copy()
        
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

        # morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
        closing = cv.morphologyEx(frame_threshold, cv.MORPH_CLOSE, kernel)
        opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

        mask = closing
        res = cv.bitwise_and(frame,frame,mask=mask)
        #cv.imshow("mask",mask)
        
        
        #blurred = cv.GaussianBlur(image_gray, (9,9), 0)
        #cv.imshow('blurred',np.hstack((frame, blurred)))
        #cv.waitKey(0)
        
        im_selected = mask

        #canny = cv.Canny(im_selected, min, max)
        #cv.imshow('canny',np.hstack((im_selected, canny)))

        ##lines = cv.HoughLinesP(frame, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]])
        #lines = cv.HoughLinesP(canny, 1, math.pi/2, 2, None, 30, 1);

        #if lines is not None:
        #    for line in lines:
        #        for x1,y1,x2,y2 in line:
        #            pt1 = (x1,y1)
        #            pt2 = (x2,y2)
        #            cv.line(frame, pt1, pt2, (0,0,255), 3)

        #cv.imshow('hough',frame)


        laplace = cv.Laplacian(im_selected, cv.CV_64F, ksize=3)
        laplace = cv.convertScaleAbs(laplace, None, 255.0/np.absolute(laplace).max())
        cv.imshow('laplace',laplace)


        sobelx = cv.Sobel(im_selected,cv.CV_64F,1,0,None,5)
        sobely = cv.Sobel(im_selected,cv.CV_64F,0,1,None,5)

        sobelx = cv.convertScaleAbs(sobelx, None, 255.0/np.absolute(sobelx).max())
        sobely = cv.convertScaleAbs(sobely, None, 255.0/np.absolute(sobely).max())

        cv.imshow('sobelx',sobelx)
        cv.imshow('sobely',sobely)

        sobel_added = sobelx + sobely
        sobel_added = cv.convertScaleAbs(sobel_added, None, 255.0/np.absolute(sobel_added).max())
        cv.imshow('sobel_added',sobel_added)

        threshold = 225

        #sobelx_bin = cv.adaptiveThreshold(sobelx,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        ret, sobelx_bin = cv.threshold(sobelx, threshold, 255, cv.THRESH_BINARY)
        #ret, sobelx_bin = cv.threshold(sobelx, threshold, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imshow('sobelx_bin', sobelx_bin)

        #sobely_bin = cv.adaptiveThreshold(sobely,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        ret, sobely_bin = cv.threshold(sobely, threshold, 255, cv.THRESH_BINARY)
        #ret, sobely_bin = cv.threshold(sobely, threshold, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imshow('sobely_bin', sobely_bin)

        #sobel_added_bin = cv.adaptiveThreshold(sobel_added,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        ret, sobel_added_bin = cv.threshold(sobel_added, threshold, 255, cv.THRESH_BINARY)
        #ret, sobel_added_bin = cv.threshold(sobel_added, threshold, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imshow('sobel_added_bin', sobel_added_bin)


        # morphology
        #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        #sobely_erosion = cv.morphologyEx(sobely_bin, cv.MORPH_OPEN, kernel)
        #cv.imshow('sobely_erosion', sobely_erosion)

        return cv.waitKey(0)
        return cv.waitKey(int(1000/30))


    def generateGaborFilters(self,lambd_iter,theta_iter,ksize,sigma):
        filters = []
        for lambd in lambd_iter:
            for theta in theta_iter:
                params = {'ksize':ksize, 'sigma':sigma, 'theta':theta, 'lambd':lambd, 'gamma':1, 'psi':0}
                kernel = cv.getGaborKernel(**params)
                kernel /= 2 * np.pi * params['sigma']**2 # in scikit_image this is done
                kernel /= kernel.sum()
                filters.append((kernel,params))
        return filters