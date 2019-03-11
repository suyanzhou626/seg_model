# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import cv2
import numpy as np
import torch
import skimage as skimg

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, is_continuous=False,fix=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size,output_size)
        else:
            self.output_size = output_size
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST
        self.fix = fix

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if self.output_size == (h,w):
            return sample
            
        if self.fix:
            h_rate = self.output_size[0]/h
            w_rate = self.output_size[1]/w
            min_rate = h_rate if h_rate < w_rate else w_rate
            new_h = h * min_rate
            new_w = w * min_rate
        else: 
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, dsize=(new_w,new_h), interpolation=cv2.INTER_CUBIC)
        
        top = (self.output_size[0] - new_h)//2
        bottom = self.output_size[0] - new_h - top
        left = (self.output_size[1] - new_w)//2
        right = self.output_size[1] - new_w - left
        if self.fix:
            img = cv2.copyMakeBorder(img,top,bottom,left,right, cv2.BORDER_CONSTANT, value=[0,0,0])  

        if 'segmentation' in sample.keys():
            segmentation = sample['segmentation'] 
            seg = cv2.resize(segmentation, dsize=(new_w,new_h), interpolation=self.seg_interpolation)
            if self.fix:
                seg = cv2.copyMakeBorder(seg,top,bottom,left,right, cv2.BORDER_CONSTANT, value=[0])
            sample['segmentation'] = seg
        sample['image'] = img
        return sample
                     
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                      left: left + new_w]

        segmentation = segmentation[top: top + new_h,
                      left: left + new_w]
        sample['image'] = image
        sample['segmentation'] = segmentation
        return sample