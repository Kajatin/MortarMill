import os
import logging
import random

import cv2 as cv
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

from . import classifier
#import classifier
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


def createTrainingData(dataset, unsupervised=True, ratio=0.05):
    X = []
    y = []

    for path in dataset:
        print(path)
        image = cv.imread(path)
        #image = cv.resize(image, (848, 480))
        #image = cv.GaussianBlur(image, (15,15), 1)

        #CF, TF = vision.imgproc.calculateColorAndTextureFeatures(image.copy())
        #features = np.hstack((CF.reshape(-1,3),TF.reshape(-1,1)))
        #features = CF.reshape(-1,3)

        if unsupervised:
            #features = image.reshape(-1,3)
            #features = cv.cvtColor(image, cv.COLOR_BGR2HSV).reshape(-1,3)
            #TODO: implement case where first all CF,TF features are calculated for all images before the FCM is called
            #training = classifier.assignLabelsUnsupervised(image, features, image.reshape(-1,3), ratio)
            training = classifier.assignLabelsUnsupervised(image, features, None, ratio)
    
            X.append(np.vstack([training[0],training[1]]))
            y.append(np.hstack([np.repeat(0,len(training[0])),np.repeat(1,len(training[1]))]))
        else:
            #features_ = features
            #features_ = image.reshape(-1,3)
            features_ = cv.cvtColor(image, cv.COLOR_BGR2HSV).reshape(-1,3)

            # select a subset of the data used for training
            selection_features_ = features_[np.random.choice(features_.shape[0],
                                                             int(features_.shape[0]*ratio),
                                                             replace=False), :]

            # determine the label from the file name 0: mortar, 1: brick
            label = int(path.split('_')[-1].split('.')[0])

            X.append(selection_features_)
            y.append(np.repeat(label,selection_features_.shape[0]))

    return np.vstack(X), np.hstack(y)


def createHsvTrainingData(dataset, ratio=0.05):
    X = []
    y = []

    for path in dataset:
        # read image data
        image = cv.imread(path)
        # convert to HSV and reshape to 2D
        features_ = cv.cvtColor(image, cv.COLOR_BGR2HSV).reshape(-1,3)
        # select a subset of the data used for training
        selection_features_ = features_[np.random.choice(features_.shape[0],
                                                         int(features_.shape[0]*ratio),
                                                         replace=False), :]
        # determine the label from the file name 0: mortar, 1: brick
        label = int(path.split('_')[-1].split('.')[0])

        X.append(selection_features_)
        y.append(np.repeat(label,selection_features_.shape[0]))

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