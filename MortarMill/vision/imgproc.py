import logging
logger = logging.getLogger(__name__)

import cv2 as cv
import numpy as np
from sklearn import linear_model
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def calculateHistogram(frame_):
    frame = frame_.copy()
    frame = np.clip(frame, 0, 1) * 255
    hist = cv.calcHist([frame.astype(np.uint8)], [0], None, [256], [0,256])

    return hist, frame


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
            r = cv.selectROI(frame)

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
                # append the index of the maximum to array
                maxs.append(hist.argmax())
                
                # plot the histogram for given channel
                plt.plot(hist,color=c)
                plt.plot(maxs[-1], hist[maxs[-1]], color=c, marker='x')
                plt.xlim([0,256])

            # show the histogram plot
            plt.show()

        else:
            logger.info('Starting HSV threshold calibration in automatic mode.')
            # automatically find brick cluster center

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
                
            ## find the cluster center corresponding to the bricks
            #mask = kmeans.labels_.reshape(frame_hsv.shape[0], frame_hsv.shape[1])

            ## create binary image where 255 corresponds to brick pixels
            #mask = np.where(mask==i, 255, 0)
            #mask = mask.astype(np.uint8)

            ## refine mask (remove small objects)
            #mask = connectedComponentsBasedFilter(mask)

            ## show mask in debug mode
            #cv.imshow('Auto HSV threshold mask', mask)

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
        eps = 10
        lowerb = tuple([int(val-eps) for val in maxs])
        upperb = tuple([int(val+eps) for val in maxs])
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
    min_size = 250

    cc_out = np.zeros((frame.shape))
    for i in range(1,nb_comp):
        if stats[i, -1] >= min_size:
            cc_out[label == i] = 255

    # connected components inverted
    inverted_image = np.zeros((cc_out.shape),np.uint8)
    inverted_image[cc_out == 0] = 255

    nb_comp_i, label_i, stats_i, centroids_i = cv.connectedComponentsWithStats(inverted_image, connectivity=8)
    min_size = 2000

    cc_out_inverted = np.zeros((frame.shape))
    for i in range(1,nb_comp_i):
        if stats_i[i,-1] >= min_size:
            cc_out_inverted[label_i == i] = 255

    # final mask
    final_mask = np.zeros((cc_out_inverted.shape),np.uint8)
    final_mask[cc_out_inverted == 0] = 255
    #cv.imshow('final mask', final_mask)

    return final_mask


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


def PCA(data, correlation = False, sort = True):
    """ Applies Principal Component Analysis to the data

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

    Notes
    -----
    The correlation matrix is a better choice when there are different magnitudes
    representing the M variables. Use covariance matrix in other cases.

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


def findBestFitPlane(points, equation=False):
    """ Computes the best fitting plane of the given points

    Parameters
    ----------        
    points: array
        The x,y,z coordinates corresponding to the points from which we want
        to define the best fitting plane. Expected format:
            array([
            [x1,y1,z1],
            ...,
            [xn,yn,zn]])

    equation(Optional) : bool
            Set the oputput plane format:
                If True return the a,b,c,d coefficients of the plane.
                If False(Default) return 1 Point and 1 Normal vector.    
    Returns
    -------
    a, b, c, d : float
        The coefficients solving the plane equation.

    or

    point, normal: array
        The plane defined by 1 Point and 1 Normal vector. With format:
        array([Px,Py,Pz]), array([Nx,Ny,Nz])

    """

    w, v = PCA(points)

    #: the normal of the plane is the last eigenvector
    normal = v[:,2]

    #: get a point from the plane
    point = np.mean(points, axis=0)


    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d

    else:
        return point, normal


def separateDepthPlanes(frame):
    h, w = frame.shape[:2]

    # ransac method
    xy = [[x,y] for x in range(h) for y in range(w)]
    z = [frame[x,y]*1000 for x in range(h) for y in range(w)]
    
    mask_ransac = np.zeros(frame.shape[:2], np.uint8)
    try:
        ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        ransac.fit(xy,z)
        mask_ransac[ransac.inlier_mask_.reshape(h,w)] = 255
    except:
        pass

    # plane fitting method with PCA
    cloud = [[x,y,frame[x,y]] for x in range(h) for y in range(w)]
                
    a,b,c,d = findBestFitPlane(cloud,True)
    boolean_mask = [(a*x+b*y+c*z+d)<0.001 for x,y,z in cloud]
    boolean_mask = np.array(boolean_mask).reshape(h,w)

    mask_plane = np.zeros(frame.shape[:2],np.uint8)
    mask_plane[boolean_mask] = 255

    #cv.imshow('mydepth',mask_plane)
    #cv.imshow('mydepth_ransac',mask_ransac)

    return mask_plane, mask_ransac