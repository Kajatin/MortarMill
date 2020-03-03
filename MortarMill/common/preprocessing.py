import cv2 as cv
import numpy as np

from common.FCM import FCM

def generateGaborFilters(lambd_iter,theta_iter,ksize,sigma):
        filters = []
        for lambd in lambd_iter:
            for theta in theta_iter:
                params = {'ksize':ksize, 'sigma':sigma, 'theta':theta, 'lambd':lambd, 'gamma':1, 'psi':0}
                kernel = cv.getGaborKernel(**params)
                kernel /= 2 * np.pi * params['sigma']**2 # in scikit_image this is done
                kernel /= kernel.sum()
                filters.append((kernel,params))
        return filters


def calculateColorAndTextureFeatures(image):
    ## Step 1: Pixel-level COLOR features extraction
    # convert color space to CIE L*a*b
    image_lab = cv.cvtColor(image,cv.COLOR_BGR2Lab)
    #image_lab = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    #image_lab = image
    #TODO: check what color space to use

    # convert dtype to float to avoid overflow during computations
    image_lab_f = image_lab.astype(float)

    # standard deviation of color component within window over image
    # sigma = sqrt(E[X^2] - E[X]^2)
    kernel_size = (15,15)
    mu = cv.blur(image_lab_f, kernel_size) # E[X]
    mu2 = cv.blur(image_lab_f**2, kernel_size) # E[X^2]
    v = cv.sqrt(mu2 - mu**2) # std of each color component

    # normalize sigma values between 0-1 for each channel
    normed_v = np.dstack([v[:,:,c]/v[:,:,c].max() for c in range(v.shape[2])])
        
    # sobel for each channel
    kernel_size = 5
    dx = cv.Sobel(image_lab_f, -1, 1, 0, ksize=kernel_size)
    dy = cv.Sobel(image_lab_f, -1, 0, 1, ksize=kernel_size)
    e = cv.sqrt(dx**2 + dy**2) # local discontinuity

    # normalize e values between 0-1 for each channel
    normed_e = np.dstack([e[:,:,c]/e[:,:,c].max() for c in range(e.shape[2])])

    # pixel level color feature (local homogeneity)
    # the more uniform a local region is, the larger the homogeneity value is
    CF = 1 - normed_e * normed_v

    ## Step 2: Pixel-level TEXTURE features extraction
    # convert color space to YCrCb
    image_ycrcb = cv.cvtColor(image,cv.COLOR_BGR2YCrCb)

    # convert dtype to float to avoid overflow during computations
    image_ycrcb_f = image_ycrcb.astype(float)
    image_y = image_ycrcb_f[:,:,0]

    # create and apply Gabor filter to luminance component of image
    filters = generateGaborFilters(
        np.linspace(3,5,3),
        #np.linspace(1,np.pi,10),
        #np.linspace(0,np.pi/2,2),
        np.linspace(0,np.pi,4),
        (15,15),
        #None,
        np.pi*3)
        #np.pi*0.75)

    # apply Gabor filter
    filtered_images = np.zeros((image_y.shape[0],image_y.shape[1],len(filters)))
    for i, filter in enumerate(filters):
        filtered_images[:,:,i] = cv.filter2D(image_y, -1, filter[0])

    # pixel texture feature extraction
    #filtered_images = cv.normalize(filtered_images, None, -1, 1, cv.NORM_MINMAX)
    filtered_images = np.abs(filtered_images)
    normed_filtered_images = np.dstack([filtered_images[:,:,c]/filtered_images[:,:,c].max() for c in range(filtered_images.shape[2])])
    TF = normed_filtered_images.max(2)


    ###################
    # will be removed
    final_color = cv.normalize(CF, None, 0, 255, cv.NORM_MINMAX)
    final_color = final_color.astype(np.uint8)
    #final_color = cv.cvtColor(final_color, cv.COLOR_Lab2BGR)
    cv.imshow('final_color',final_color)

    final_texture = cv.normalize(TF, None, 0, 255, cv.NORM_MINMAX)
    final_texture = final_texture.astype(np.uint8)
    cv.imshow('final_texture',final_texture)

    # visualize filtered images
    for i in range(filtered_images.shape[2]):
        filter_to_show = cv.normalize(filtered_images[:,:,i], None, 0, 255, cv.NORM_MINMAX)
        filter_to_show = filter_to_show.astype(np.uint8)
        cv.imshow(f'filtered_image_{i}',filter_to_show)
    
    # visualize filters
    for i,filter in enumerate(filters):
        filter_to_show = cv.resize(filter[0],(200,200))
        filter_to_show = cv.normalize(filter_to_show, None, 0, 255, cv.NORM_MINMAX)
        filter_to_show = filter_to_show.astype(np.uint8)
        cv.imshow(f'filter_{i}',filter_to_show)


    return CF, TF


def assignLabelsUnsupervised(image, features, features_t=None, ratio=0.05):
    #TODO: remove image input, not needed

    if features_t is None:
        features_t = features

    # fuzzy c-means
    # fit the fuzzy-c-means
    fcm = FCM(n_clusters=2,m=1.3,max_iter=1500,error=1e-7,random_state=np.random.randint(10000))
    fcm.fit(features)

    # outputs
    fcm_centers = fcm.centers
    fcm_labels  = fcm.u.argmax(axis=1)
    unique, counts = np.unique(fcm_labels, return_counts=True)
    print(fcm_centers)
    print(unique, counts)

    fcm_labels = np.ones((features.shape[0],)) * 0.5
    training = {}
    keys = unique
    for i in unique:
        args = fcm.u.argpartition(-int(counts[i]*ratio),axis=0)[-int(counts[i]*ratio):]
        fcm_labels[args[:,i]] = i
        training[keys[i]] = features_t[args[:,i],:]

    fcm_labels = fcm_labels.reshape(image.shape[:2])
    fcm_labels = cv.normalize(fcm_labels, None, 0, 255, cv.NORM_MINMAX)
    fcm_labels = fcm_labels.astype(np.uint8)
    cv.imshow('fcm_labels',fcm_labels)

    fcm_labels  = fcm.u.argmax(axis=1)
    fcm_labels = fcm_labels.reshape(image.shape[:2])
    fcm_labels = cv.normalize(fcm_labels, None, 0, 255, cv.NORM_MINMAX)
    fcm_labels = fcm_labels.astype(np.uint8)
    cv.imshow('fcm_labels_orig',fcm_labels)

    return training