import logging
logger = logging.getLogger(__name__)

import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def movingAverage(data, window_size=5):
    """ Calculates the moving average of the input array for the given window size.

    Parameters
    ----------        
    data: array
        The input array (1D) for which the moving average is calculated.

    windows_size: int (default: 5)
        The size of the moving windows used for the averaging. The larger this
        parameters is, the smoother the output becomes.

    Returns
    -------
    ret: array
        The averaged version of the input array. The size of this array is smaller
        by windows_size-1.
    """
    
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def calibrateHsvThresholds(frame, supervised=False):
    """
    Finds the best HSV threshold values used to separate the bricks from
    the mortar.

    Parameters
    ----------        
    frame: array
        The array containing the colour image data. The input is interpreted
        as a BGR 3-channel image. If `frame` hasn't got 3 channels, this
        function fails and returns None.

    supervised (Optional): bool (Default: False)
        If set to True, the HSV calibration is performed manually by the user.
        During the process, the user selects the area of the brick. Once the
        calibration is done, the user can press `s` to accept the results,
        `esc` to cancel the calibration, and any other button to try again.
        If False, the calibration is done automatically. The algorithm finds
        the suitable HSV threshold values that separate the bricks from the mortar.

    Returns
    -------
    lowerb, upperb: tuple, tuple
        Returns the lower and upper 3 HSV threshold values if the calibration is
        successful, otherwise returns None.
    """

    # validate the arguments
    if frame is None:
        logger.warning(('The HSV threshold calibration is unsuccessful because '
                        'the input frame is None.'))
        return None

    if len(frame.shape) != 3 or frame.shape[2] != 3:
        logger.warning(('The input `frame` does not have the correct number of '
                        'channels. The HSV threshold calibration is unsuccessful.'))
        return None

    while 1:
        if supervised:
            logger.info('Starting HSV threshold calibration in manual mode.')

            # manually select ROI (area of a brick)
            r = cv.selectROI('Select brick area', frame)

            # crop image -> extract the brick pixels
            frame_cropped = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            # convert to HSV
            frame_hsv = cv.cvtColor(frame_cropped, cv.COLOR_BGR2HSV)
            
            # evaluate histograms
            maxs = []
            colours = ('b','g','r')
            for i,c in enumerate(colours):
                # calculate histogram for given channel
                hist = cv.calcHist([frame_hsv], [i], None, [256], [0,256])
                # smooth out histogram with moving average
                hist = movingAverage(hist.ravel()).reshape(-1,1)
                # append the index of the maximum to array
                maxs.append(hist.argmax())
                # plot the histogram for given channel
                plt.plot(hist,color=c)
                plt.plot(maxs[-1], hist[maxs[-1]], color=c, marker='x')
                plt.xlim([0,256])

            # show the histogram plot
            plt.show()

        else:
            # automatically find brick cluster center
            logger.info('Starting HSV threshold calibration in automatic mode.')

            # convert to HSV
            frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            # normalize values to 0-1 range
            frame_hsv = frame_hsv.astype(np.float64)
            frame_hsv[:,:,0] /= 179.0
            frame_hsv[:,:,1:3] /= 255.0

            # reshape image into 2D array
            frame_to_fit = frame_hsv.reshape(-1, 3)

            # fit kmeans with a number of classes on the array
            kmeans = KMeans(n_clusters=4).fit(frame_to_fit)

            #TODO: refine this code to find the correct cluster center
                
            # determine the count of unique kmeans labels assuming that the
            # label belonging to the bricks will be most frequent
            unique, count = np.unique(kmeans.labels_, return_counts=True)

            # select the index of most frequent unique label (these are the bricks)
            i = unique[count.argmax()]
                
            # DEBUG *******************************************
            # find the cluster center corresponding to the bricks
            mask = kmeans.labels_.reshape(frame_hsv.shape[0], frame_hsv.shape[1])
            # normalize mask to 0-255 range
            mask = mask.astype(float)
            mask *= 255/mask.max()
            mask = np.abs(mask-255)
            mask = mask.astype(np.uint8)
            # create binary image where 255 corresponds to brick pixels
            #mask = np.where(mask==i, 255, 0)
            # refine mask (remove small objects)
            #mask = connectedComponentsBasedFilter(mask)
            # show mask in debug mode
            #cv.imshow('input',frame)
            #cv.imshow('Auto HSV threshold mask', mask)
            #cv.imwrite('hsv_colour_kmeans_original.png',frame)
            #cv.imwrite('hsv_colour_kmeans_clusters.png',mask)
            # DEBUG *******************************************

            logger.debug(('Found cluster centers for HSV thresholds: {} . '
                          'The selected cluster index is {}.')
                          .format(kmeans.cluster_centers_, i))
                
            # set the `maxs` array with the HSV threshold values
            maxs = kmeans.cluster_centers_[i].copy()

            # scale back up to the correct HSV range
            maxs[0] *= 179
            maxs[1:3] *= 255
            maxs = maxs.astype(np.uint8)

            logger.info('Final HSV threshold values: {}'.format(maxs))

        # generate lower and upper HSV bounds
        epss = [10, 60, 60]
        lowerb = tuple([int(val-eps) for val, eps in zip(maxs,epss)])
        upperb = tuple([int(val+eps) for val, eps in zip(maxs,epss)])
        logger.debug(('Candidate HSV threshold ranges found (not yet '
                      'accepted). Lower: {} Upper: {}').format(lowerb,upperb))

        if supervised:
            # test the HSV thresholding
            frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            frame_threshold = cv.inRange(frame_HSV, lowerb, upperb)
            cv.imshow('threshold', frame_threshold)

            key = cv.waitKey(0)
            if key == 27:
                logger.warning(('Calibration of the HSV threshold ranges is unsuccessful. '
                                'The user did not accept the any of the results.'))
                return None
            elif key != ord('s'):
                continue

        # save the HSV threshold calibration results
        logger.info(('Calibrated the HSV threshold ranges. Lower: {}\tupper: {}')
                     .format(lowerb, upperb))
        break

    #cv.destroyAllWindows()
    return lowerb, upperb


def connectedComponentsBasedFilter(frame):
    if len(frame.shape) != 2:
        logger.error('This function can only be called with a single channel 8-bit image.')
        return None

    # connected components
    nb_comp, label, stats, centroids = cv.connectedComponentsWithStats(frame, connectivity=8)

    # keep large areas
    cc_out = np.full_like(frame,0)
    min_size = 250
    for i in range(1,nb_comp):
        if stats[i, -1] >= min_size:
            cc_out[label == i] = 255

    # connected components inverted
    inverted_image = np.invert(cc_out)
    nb_comp_i, label_i, stats_i, centroids_i = cv.connectedComponentsWithStats(inverted_image, connectivity=8)

    # keep large areas
    cc_out_inverted = np.full_like(frame,0)
    min_size = 2000
    for i in range(1,nb_comp_i):
        if stats_i[i,-1] >= min_size:
            cc_out_inverted[label_i == i] = 255

    # final mask
    return np.invert(cc_out_inverted)


def calculateColorAndTextureFeatures(image):
    ## Step 1: Pixel-level COLOR features extraction
    # convert color space to CIE L*a*b
    image_lab = cv.cvtColor(image,cv.COLOR_BGR2Lab)

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
        (9,9),
        #None,
        #np.pi)
        np.pi*0.95)

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
    ## will be removed
    #final_color = cv.normalize(CF, None, 0, 255, cv.NORM_MINMAX)
    #final_color = final_color.astype(np.uint8)
    ##final_color = cv.cvtColor(final_color, cv.COLOR_Lab2BGR)
    #cv.imshow('final_color',final_color)

    #final_texture = cv.normalize(TF, None, 0, 255, cv.NORM_MINMAX)
    #final_texture = final_texture.astype(np.uint8)
    #cv.imshow('final_texture',final_texture)

    ## visualize filtered images
    #for i in range(filtered_images.shape[2]):
    #    filter_to_show = cv.normalize(filtered_images[:,:,i], None, 0, 255, cv.NORM_MINMAX)
    #    filter_to_show = filter_to_show.astype(np.uint8)
    #    cv.imshow(f'filtered_image_{i}',filter_to_show)
    
    ## visualize filters
    #for i,filter in enumerate(filters):
    #    filter_to_show = cv.resize(filter[0],(200,200))
    #    filter_to_show = cv.normalize(filter_to_show, None, 0, 255, cv.NORM_MINMAX)
    #    filter_to_show = filter_to_show.astype(np.uint8)
    #    cv.imshow(f'filter_{i}',filter_to_show)

    return CF, TF


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


def PCA(data, correlation=False, sort=True):
    """ Applies Principal Component Analysis to the data.

    Parameters
    ----------        
    data: array
        The array containing the data. The array must have NxM dimensions, where each
        of the N rows represents a different individual record and each of the M columns
        represents a different variable recorded for that individual record.
            array([
            [V11, ... , V1m],
            ...,
            [Vn1, ... , Vnm]])

    correlation(Optional) : bool
            Set the type of matrix to be computed (see Notes):
                If True compute the correlation matrix.
                If False(Default) compute the covariance matrix. 

    sort(Optional) : bool
            Set the order that the eigenvalues/vectors will have
                If True(Default) they will be sorted (from higher value to less).
                If False they won't.   
    Returns
    -------
    eigenvalues: (1,M) array
        The eigenvalues of the corresponding matrix.

    eigenvector: (M,M) array
        The eigenvectors of the corresponding matrix.

    """

    mean = np.mean(data, axis=0)

    data_adjust = data - mean

    #: the data is transposed due to np.cov/corrcoef syntax
    if correlation:

        matrix = np.corrcoef(data_adjust.T)

    else:
        matrix = np.cov(data_adjust.T) 

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]

    return eigenvalues, eigenvectors


def separateDepthPlanes(frame):
    """ Separates the depth map input into two labels based on how well each point
    fits the best-fit plane. The best-fit plane is found by applying PCA on the
    point cloud converted input.

    Parameters
    ----------        
    frame: array
        The array containing the depth data. The input is interpreted
        as a single channel image.

    Returns
    -------
    mask: array
        The mask image array (1 channel, binary). 255 represents brick
        pixels, 0 for the mortar.
    """

    # remove 0s, upper and lower 0.5% outliers
    lower_percentile = np.percentile(frame,0.5)
    upper_percentile = np.percentile(frame,99.5)
    idxs = np.where((frame>lower_percentile) & (frame < upper_percentile) & (frame != 0))
    cloud_filtered = np.dstack([np.dstack(idxs),frame[idxs]*1000]).squeeze()
    
    # plane fitting method with PCA
    w, v = PCA(cloud_filtered)
    normal = v[:,2]
    point = np.mean(cloud_filtered, axis=0)
    a, b, c = normal
    d = -(np.dot(normal, point))

    # evaluate how well each point fits the plane
    y,x = np.meshgrid(np.arange(frame.shape[1]),np.arange(frame.shape[0]))
    cloud = np.dstack([x.flatten(),y.flatten(),frame.flatten()*1000]).squeeze()
    boolean_mask = (np.sum(cloud * np.array([a,b,c]), 1) + d) < 1
    boolean_mask = np.array(boolean_mask).reshape(frame.shape)
    boolean_mask[frame==0] = False

    # create mask
    mask = np.zeros(frame.shape,np.uint8)
    mask[boolean_mask] = 255

    return mask


def undistortFrame(frame, params):
    if frame is None:
        logger.warning('Cannot do undistortion because the `frame` argument is None.')
        return

    if params is None:
        logger.warning('Cannot do undistortion because the `params` argument is None.')
        return

    if isinstance(params, dict):
        # extract the camera parameters for the colour image
        camera_matrix = params['intrinsics']
        distortion_coeffs = params['distortion']
        new_camera_matrix = params['new_cam_matrix']
        roi = params['roi']

        logger.debug('Undistorting frame using provided camera parameters.')
        logger.debug('Camera parameters: {}'.format(params))

        # undistort image
        frame = cv.undistort(frame, camera_matrix, distortion_coeffs, None, new_camera_matrix)

        # crop the image
        x, y, w, h = roi
        #frame = frame[y:y+h, x:x+w]

    else:
        # built-in calib params
        camera_matrix = np.array([[params.fx, 0, params.ppx],
                                  [0, params.fy, params.ppy],
                                  [0, 0, 1]])
        distortion_coeffs = np.array([params.coeffs])
        
        logger.debug('Undistorting frame using built-in camera parameters.')
        logger.debug('Camera parameters: {}'.format(params))

        # undistort image
        frame = cv.undistort(frame, camera_matrix, distortion_coeffs, None)

    return frame


def normalise(frame, mode='hsv'):

    frame_ = np.float64(frame)

    if mode == 'rgb':
        frame_ = frame / 255.0
    elif mode == 'hsv':
        frame_[:,:,0] = frame_[:,:,0] / 179.0
        frame_[:,:,1:3] = frame_[:,:,1:3] / 255.0
    elif mode == 'depth':
        frame_ = frame / 0.246
    else:
        return None

    return frame_


def skeletonize(mask):
    from skimage.morphology import skeletonize, medial_axis
    skeleton1 = skeletonize(np.invert(mask)/255.0)
    skeleton = medial_axis(np.invert(mask)/255.0)
    cv.imshow('inverse mask',np.invert(mask))
    cv.imshow('Skeleton1', (skeleton1*255).astype(np.uint8))
    cv.imshow('Skeleton', (skeleton*255).astype(np.uint8))

    data = np.invert(mask)/255.0
    # Compute the medial axis (skeleton) and the distance transform
    skel, distance = medial_axis(data, return_distance=True)
    # Distance to the background for pixels of the skeleton
    dist_on_skel = distance * skel
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(dist_on_skel, cmap=plt.cm.Spectral, interpolation='nearest')
    plt.contour(data, [0.5], colors='w')
    plt.axis('off')
    plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    plt.show()


def visualizeFinalSegmentation(frame, mask):
    # detect bricks with final mask
    frame_ = frame.copy()
    nb_comp_f, label_f, stats_f, centroids_f = cv.connectedComponentsWithStats(mask, connectivity=8)
    for center in centroids_f[1:]:
        cv.drawMarker(frame_, tuple(np.uint(center)), (255,0,0), cv.MARKER_CROSS, thickness=2)

    for stat in np.uint(stats_f[1:]):
        cv.rectangle(frame_,
                    (stat[0],stat[1]),(stat[0]+stat[2],stat[1]+stat[3]),
                    (0,0,255), 2)

    # detect contours
    cnts, hrcy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame_, cnts, -1, (0,255,0),2)
    
    # show brick detection results
    masked_orig = cv.bitwise_and(frame,frame,mask=np.invert(mask))
    cv.imshow('Detected bricks',np.vstack((frame_,masked_orig)))
