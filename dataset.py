from __future__ import print_function

import numpy as np
from skimage import color
from torchvision.transforms import transforms
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import Dataset

class ImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, target, index


class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img


class RGB2HSV(object):
    """Convert RGB PIL image to ndarray HSV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hsv(img)
        return img


class RGB2HED(object):
    """Convert RGB PIL image to ndarray HED."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hed(img)
        return img


class RGB2LUV(object):
    """Convert RGB PIL image to ndarray LUV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2luv(img)
        return img


class RGB2YUV(object):
    """Convert RGB PIL image to ndarray YUV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yuv(img)
        return img


class RGB2XYZ(object):
    """Convert RGB PIL image to ndarray XYZ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2xyz(img)
        return img


class RGB2YCbCr(object):
    """Convert RGB PIL image to ndarray YCbCr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ycbcr(img)
        return img


class RGB2YDbDr(object):
    """Convert RGB PIL image to ndarray YDbDr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ydbdr(img)
        return img


class RGB2YPbPr(object):
    """Convert RGB PIL image to ndarray YPbPr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ypbpr(img)
        return img


class RGB2YIQ(object):
    """Convert RGB PIL image to ndarray YIQ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yiq(img)
        return img


class RGB2CIERGB(object):
    """Convert RGB PIL image to ndarray RGBCIE."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2rgbcie(img)
        return img

import pandas as pd
class BioInfoDataset(Dataset):
    """数据集演示"""
    def __init__(self, data_name):
        """实现初始化方法，在初始化的时候将数据载入"""
        self.x = np.load('data/scRNAseq/exported/raw-' + data_name + '-X.npy')
        y = pd.read_csv('data/scRNAseq/exported/raw-' + data_name + '-labels.csv', index_col=0).index
        labels = y.unique()
        self.y = np.array([labels == l for l in y]).astype(np.float32).argmax(-1)
        self.classes = list(labels)

    def __len__(self):
        '''
        返回df的长度
        '''
        return len(self.x)
    def __getitem__(self, idx):
        '''
        根据idx返回一行数据
        '''
        return self.x[idx], self.y[idx]


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class NoAugTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, augment):
        self.base_transform = base_transform
        self.augment = augment

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.augment(x)
        return [q, k]

class MultiViewTransform(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class TwoSameTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        #k = self.base_transform(x)
        return [q, q]

import pandas as pd
class BioInfoDataset(Dataset):
    """数据集演示"""
    def __init__(self, data_name='Lake_2018'):# [Lake_2018, Campbell, Chen, Tasic18, Tosches_lizard, Tosches_turtle]
        """实现初始化方法，在初始化的时候将数据载入"""
        assert data_name in ['Lake_2018', 'Campbell', 'Chen', 'Tasic18', 'Tosches_lizard', 'Tosches_turtle'], 'wrong data name!'
        self.x = np.load('../data/scRNAseq/exported/raw-' + data_name + '-X.npy')
        y = pd.read_csv('../data/scRNAseq/exported/raw-' + data_name + '-labels.csv', index_col=0).index
        assert len(y) == len(self.x), 'the length can not match!'
        labels = y.unique()
        self.y = np.array([labels == l for l in y]).astype(np.float32).argmax(-1)
        self.classes = list(labels)

    def __len__(self):
        '''
        返回df的长度
        '''
        return len(self.x)
    def __getitem__(self, idx):
        '''
        根据idx返回一行数据
        '''
        return self.x[idx], self.y[idx]


class IndexedBioInfoDataset(Dataset):
    """数据集演示"""
    def __init__(self, data_name='Lake_2018'):# [Lake_2018, Campbell, Chen, Tasic18, Tosches_lizard, Tosches_turtle]
        """实现初始化方法，在初始化的时候将数据载入"""
        assert data_name in ['Lake_2018', 'Campbell', 'Chen', 'Tasic18', 'Tosches_lizard', 'Tosches_turtle'], 'wrong data name!'
        self.x = np.load('../data/scRNAseq/exported/raw-' + data_name + '-X.npy')
        y = pd.read_csv('../data/scRNAseq/exported/raw-' + data_name + '-labels.csv', index_col=0).index
        assert len(y) == len(self.x), 'the length can not match!'
        labels = y.unique()
        self.y = np.array([labels == l for l in y]).astype(np.float32).argmax(-1)
        self.classes = list(labels)

    def __len__(self):
        '''
        返回df的长度
        '''
        return len(self.x)
    def __getitem__(self, idx):
        '''
        根据idx返回一行数据
        '''
        return self.x[idx], self.y[idx], idx
