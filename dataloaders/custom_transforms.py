import torch
import random
import numpy as np
import skimage as skimg
import cv2
from PIL import Image, ImageOps, ImageFilter

class GaussianNoise(object):
    def __init__(self,mean=0,std=0):
        self.mean = mean
        self.std = std
    def __call__(self,sample):
        img = sample['image']
        sample['image'] = skimg.util.random_noise(img,mode='gaussian',mean=self.mean,var=(self.std)**2)

        return sample

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.),std=1.0):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        img = sample['image']
        img -= self.mean
        img /= self.std
        sample['image'] = img

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = img.transpose((2, 0, 1)).copy()
        mask = mask.copy()

        sample['image'] = torch.from_numpy(img).float()
        sample['label'] = torch.from_numpy(mask).float()

        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, segmentation = sample['image'], sample['label']
        if np.random.rand() < 0.5:
            image_flip = np.flip(image, axis=1)
            segmentation_flip = np.flip(segmentation, axis=1)
            sample['image'] = image_flip
            sample['label'] = segmentation_flip
        return sample


class RandomRotate(object):
    """Randomly rotate image"""
    def __init__(self, angle_r, is_continuous=False):
        self.angle_r = angle_r
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['label']
        row, col, _ = image.shape
        rand_angle = np.random.randint(-self.angle_r, self.angle_r) if self.angle_r != 0 else 0
        m = cv2.getRotationMatrix2D(center=(col/2, row/2), angle=rand_angle, scale=1)
        new_image = cv2.warpAffine(image, m, (col,row), flags=cv2.INTER_CUBIC, borderValue=0)
        new_segmentation = cv2.warpAffine(segmentation, m, (col,row), flags=self.seg_interpolation, borderValue=0)
        sample['image'] = new_image
        sample['label'] = new_segmentation
        return sample

class Resize(object):
    def __init__(self,output_size,shrink=16,is_continuous=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size,output_size)
        else:
            self.output_size = output_size
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST
        self.shrink = shrink
    
    def __call__(self,sample):
        img = sample['image']
        mask = sample['label']
        h, w ,_ = img.shape
        scale = min(self.output_size[1]/float(w),self.output_size[0]/float(h))
        out_w = int(w*scale)
        out_h = int(h*scale)
        out_w = ((out_w - 1 + self.shrink -1) // self.shrink) * self.shrink +1
        out_h = ((out_h - 1 + self.shrink -1) // self.shrink) * self.shrink +1
        img = cv2.resize(img, dsize=(out_w,out_h), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask,dsize=(out_w,out_h), interpolation=self.seg_interpolation)
        sample['image'] = img
        sample['label'] = mask
        return sample

class RandomScale(object):
    def __init__(self,rand_resize,is_continuous=False):
        self.rand_resize = [0.75,1.25] if rand_resize is None else rand_resize
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        h, w ,_ = img.shape
        rand_scale = random.uniform(self.rand_resize[0], self.rand_resize[1])
        img = cv2.resize(img, None, fx=rand_scale, fy=rand_scale, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
        sample['image'] = img
        sample['label'] = mask

        return sample

class RandomCrop(object):
    def __init__(self,crop_size):
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size

    def __call__(self,sample):
        img = sample['image']
        mask = sample['label']
        h, w ,_ = img.shape
        new_h, new_w = self.crop_size
        new_img = np.zeros((new_h,new_w,3),dtype=np.float)
        new_mask = np.zeros((new_h,new_w),dtype=np.float)
        padw = max(0,w-new_w)
        padh = max(0,h-new_h)
        w_begin = random.randint(0,padw)
        h_begin = random.randint(0,padh)
        w_end = w_begin + min(w,new_w)
        h_end = h_begin + min(h,new_h)
        new_img[0:min(h,new_h),0:min(w,new_w)] = img[h_begin:h_end,w_begin:w_end]
        new_mask[0:min(h,new_h),0:min(w,new_w)] = mask[h_begin:h_end,w_begin:w_end]
        sample['image'] = new_img
        sample['label'] = new_mask
        return sample

def onehot(label, num):
    m = label
    one_hot = np.eye(num)[m]
    return one_hot

class Multiscale(object):
    def __init__(self, rate_list):
        self.rate_list = rate_list

    def __call__(self, sample):
        image = sample['image']
        row, col, _ = image.shape
        image_multiscale = []
        for rate in self.rate_list:
            rescaled_image = cv2.resize(image, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
            sample['image_%f'%rate] = rescaled_image
        return sample