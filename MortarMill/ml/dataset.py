import os
import logging
import random

import cv2 as cv
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

import vision


logger = logging.getLogger(__name__)


def loadDatasetPath(basepath):
    """ Creates a list of filenames that comprise the dataset.

    Parameters
    ----------
    basepath: string, list
        String that specifies the path to the dataset files.
        If it is a string, it is assumed to be the path to the directory which
        contains the dataset files. If it is a list, it is assumed to contain
        the filenames used to create the dataset.

    Returns
    -------
    dataset: list
        Returns a list of filenames that comprise the dataset. If the input was
        invalid, return None.
    """

    dataset = []

    # if the input is a list, it is assumed to contain the filenames to be used
    # to create the dataset
    if isinstance(basepath, list):
        logger.debug(('Provided path argument is interpreted to contain the '
                     'filenames used to create the dataset. ({})').format(basepath))
        for path in basepath:
            if os.path.isfile(path):
                dataset.append(path)
                logger.debug('Adding {} to dataset.'.format(path))
    # if the input is a string, it is assumed to be the path to a directory which
    # contains all the files used to create the dataset
    elif isinstance(basepath, str):
        logger.debug(('Provided path argument is interpreted to contain the '
                     'directory that contains the dataset files.'
                     ' ({})').format(basepath))
        for entry in os.listdir(basepath):
            if not entry.startswith('.'):
                path = os.path.join(basepath,entry)
                if os.path.isfile(path):
                    dataset.append(path)
                    logger.debug('Adding {} to dataset.'.format(path))
    # otherwise, fail to create the dataset
    else:
        logger.error(('Cannot create dataset, because the input ({}) was '
                      'incorrect.'.format(basepath)))

    logger.debug('Returning dataset: {}'.format(dataset))
    # if the dataset is empty return None
    return dataset if len(dataset) > 0 else None


def assignLabelsUnsupervised(image, features, ratio=0.05):
    #TODO: remove image input, not needed
    
    # fuzzy c-means
    # fit the fuzzy-c-means
    fcm = vision.FCM(n_clusters=2,m=1.3,max_iter=1500,error=1e-7,random_state=np.random.randint(10000))
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
        training[keys[i]] = features[args[:,i],:]

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


def createTrainingData(dataset, ratio=0.05, mode='labelled', feature='rgb'):
    """ Creates a classifier dataset from the `dataset`. Supports multiple labelling
    methods and feature spaces.

    Parameters
    ----------
    dataset: array
        Array of filenames used to generate the training dataset.
    ratio: float (default: 0.05)
        Ratio of the total data used to create a training set.
    mode: string (default: 'labelled')
        Must be one of 'labelled', 'manual', or 'unsupervised'. In labelled mode,
        the filenames must contain either a 0 or a 1 to indicate which class the
        examples belong to in a given image (e.g. brick_1.png). 0 indicates mortar
        and 1 indicates bricks. In manual mode, the user needs to select an area of
        interest from each input image. In unsupervised mode, a FCM algorithm is
        used to label the images and the top x percent of most likely class members
        are used to generate the training set.
    feature: string (default: 'rgb')
        Must be one of 'rgb', 'hsv', or 'colour-texture'. Indicates the training
        data's nature. If 'colour-texture' is supplied, an algorithm produces
        colour and texture features. Later on, classifiers need to be supplied the
        same type of data for predictions.
        
    Returns
    -------
    (X, y): tuple
        Tuple of X and y, the training set and the labels (each of which are numpy arrays).
    """

    logger.debug('Creating training data with {} features and {} mode.'.format(feature,mode))
    
    X = []
    y = []

    for path in dataset:
        image = cv.imread(path)
        #image = cv.resize(image, (848, 480))
        #image = cv.GaussianBlur(image, (15,15), 1)

        if feature == 'rgb':
            features = image.reshape(-1,3)
        elif feature == 'hsv':
            features = cv.cvtColor(image, cv.COLOR_BGR2HSV).reshape(-1,3)
        elif feature == 'colour-texture':
            CF, TF = vision.imgproc.calculateColorAndTextureFeatures(image.copy())
            features = np.hstack((CF.reshape(-1,3),TF.reshape(-1,1)))
        else:
            logger.error(('The provided `feature` is not supported ({}). It should '
                    'either be \'rgb\', \'hsv\', or \'colour-texture\'.')\
                        .format(feature))

        if mode == 'labelled':
            # select a subset of the data used for training
            selection_features_ = features[np.random.choice(features.shape[0],
                                                            int(features.shape[0]*ratio),
                                                            replace=False), :]

            # determine the label from the file name 0: mortar, 1: brick
            label = int(path.split('_')[-1].split('.')[0])

            X.append(selection_features_)
            y.append(np.repeat(label,selection_features_.shape[0]))

        elif mode == 'manual':
            # manually select ROI of brick to generate training data
            r = cv.selectROI('Select brick area',image)
            cv.destroyWindow('Select brick area')
            # crop feature
            features_ = features.reshape(image.shape)
            frame_cropped = features_[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            brick = frame_cropped.reshape(-1,3)
            brick_y = np.ones(brick.shape[0])

            # manually select ROI of mortar to generate training data
            r = cv.selectROI('Select mortar area',image)
            cv.destroyWindow('Select mortar area')
            # crop image
            frame_cropped = features_[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            mortar = frame_cropped.reshape(-1,3)
            mortar_y = np.zeros(mortar.shape[0])

            X.append(np.vstack([brick,mortar]))
            y.append(np.hstack([brick_y,mortar_y]))

        elif mode == 'unsupervised':
            #TODO: implement case where first all CF,TF features are calculated for all images before the FCM is called
            training = assignLabelsUnsupervised(image, features, ratio)
    
            X.append(np.vstack([training[0],training[1]]))
            y.append(np.hstack([np.repeat(0,len(training[0])),np.repeat(1,len(training[1]))]))

        else:
            logger.error(('The provided `mode` is not supported ({}). It should '
                          'either be \'manual\', \'labelled\', or \'unsupervised\'.')\
                              .format(mode))
            return None

    return np.vstack(X), np.hstack(y)


#TODO: code this function properly (getRandomTestImage())
def getRandomTestImage():
    image = cv.imread('samples/RAW/brick_zoom_3.jpg')
    #image = cv.imread('samples/flower2.jpg')
    return cv.resize(image, (848, 480))


class BrickDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale

        self.ids = loadDatasetPath(imgs_dir)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = self.ids[idx]
        mask_file = self.ids[idx].replace(self.imgs_dir,self.masks_dir)

        img = Image.open(img_file)
        mask = Image.open(mask_file)

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}


def generateUnetDataset():
    count = 0
    dataset = loadDatasetPath('samples/large/images')

    for f in dataset:
        try:
            image = Image.open(f)
            mask = Image.open(f.replace('images','masks'))

            if image.mode != 'RGB':
                image = image.convert('RGB')
            if mask.mode != 'L':
                mask = mask.convert('L')

            # w, h = 106, 60
            w, h = 848, 480
            num = 20
            r_col = [random.randint(0,image.size[0]-w) for _ in range(num)]
            r_row = [random.randint(0,image.size[1]-h) for _ in range(num)]

            # indexing is: column, row
            for r_c, r_r in zip(r_col,r_row):
                box = (r_c,r_r,r_c+w,r_r+h)
                image_cropped = image.crop(box)
                mask_cropped = mask.crop(box)

                if random.random() < 0.3:
                    i = random.choice(range(4))
                    if i == 0:
                        image_cropped = image_cropped.transpose(Image.FLIP_LEFT_RIGHT)
                        mask_cropped = mask_cropped.transpose(Image.FLIP_LEFT_RIGHT)
                    elif i == 1:
                        image_cropped = image_cropped.transpose(Image.FLIP_TOP_BOTTOM)
                        mask_cropped = mask_cropped.transpose(Image.FLIP_TOP_BOTTOM)
                    elif i == 2:
                        enhancer = ImageEnhance.Contrast(image_cropped)
                        if random.random() < 0.5:
                            image_cropped = enhancer.enhance(1-random.randint(0,20)/100)
                        else:
                            image_cropped = enhancer.enhance(1+random.randint(0,20)/100)
                    elif i == 3:
                        enhancer = ImageEnhance.Brightness(image_cropped)
                        if random.random() < 0.5:
                            image_cropped = enhancer.enhance(1-random.randint(0,20)/100)
                        else:
                            image_cropped = enhancer.enhance(1+random.randint(0,20)/100)

                image_cropped.save(f'samples/images/image_{count}.png')
                mask_cropped.save(f'samples/masks/image_{count}.png')
                count += 1
        except FileNotFoundError:
            pass