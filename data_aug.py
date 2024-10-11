from __future__ import absolute_import

import numpy as np
import torch
from torchvision.transforms import *

#from PIL import Image
import random
import math
#import numpy as np
#import torch


class ChannelAdap(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5):
        self.probability = probability

       
    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
            # return img

        idx = random.randint(0, 3)
        
        if idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            img = img

        return img
        
        
class ChannelAdapGray(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5):
        self.probability = probability

       
    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
            # return img

        idx = random.randint(0, 3)
        
        if idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            if random.uniform(0, 1) > self.probability:
                # return img
                img = img
            else:
                tmp_img = 0.2989 * img[0,:,:] + 0.5870 * img[1,:,:] + 0.1140 * img[2,:,:]
                img[0,:,:] = tmp_img
                img[1,:,:] = tmp_img
                img[2,:,:] = tmp_img
        return img

class ChannelRandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img


class ChannelChoose(object):
    def __init__(self, gray=3):
        self.gray = gray

    def __call__(self, img):

        while True:
            idx1 = random.randint(0, self.gray)
            idx2 = random.randint(0, self.gray)
            idx3 = random.randint(0, self.gray)

            if (idx1, idx2, idx3) != (0, 1, 2):
                break

        # if idx == 0:
        #     # random select R Channel
        #     img[1, :, :] = img[0, :, :]
        #     img[2, :, :] = img[0, :, :]
        # elif idx == 1:
        #     # random select B Channel
        #     img[0, :, :] = img[1, :, :]
        #     img[2, :, :] = img[1, :, :]
        # elif idx == 2:
        #     # random select G Channel
        #     img[0, :, :] = img[2, :, :]
        #     img[1, :, :] = img[2, :, :]
        # else:
        #     # tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
        #     img[0, :, :] = img[0, :, :]
        #     img[1, :, :] = img[1, :, :]
        #     img[2, :, :] = img[2, :, :]

        img1 = img
        tmp_img = 0.2989 * img1[0, :, :] + 0.5870 * img1[1, :, :] + 0.1140 * img1[2, :, :]
        # if idx1 == 0:
        #     img[0, :, :] = img1[1, :, :]
        # elif idx1 == 1:
        #     img[0, :, :] = img1[2, :, :]
        # else:
        #     img[0, :, :] = tmp_img
        #
        # if idx2 == 0:
        #     img[1, :, :] = img1[0, :, :]
        # elif idx2 == 1:
        #     img[1, :, :] = img1[2, :, :]
        # else:
        #     img[1, :, :] = tmp_img
        #
        # if idx3 == 0:
        #     img[2, :, :] = img1[0, :, :]
        # elif idx3 == 1:
        #     img[2, :, :] = img1[1, :, :]
        # else:
        #     img[2, :, :] = tmp_img

        if idx1 == 0:
            img[0, :, :] = img1[0, :, :]
        elif idx1 == 1:
            img[0, :, :] = img1[1, :, :]
        elif idx1 == 2:
            img[0, :, :] = img1[2, :, :]
        else:
            img[0, :, :] = tmp_img

        if idx2 == 0:
            img[1, :, :] = img1[0, :, :]
        elif idx2 == 1:
            img[1, :, :] = img1[1, :, :]
        elif idx2 == 2:
            img[1, :, :] = img1[2, :, :]
        else:
            img[1, :, :] = tmp_img

        if idx3 == 0:
            img[2, :, :] = img1[0, :, :]
        elif idx3 == 1:
            img[2, :, :] = img1[1, :, :]
        elif idx3 == 2:
            img[2, :, :] = img1[2, :, :]
        else:
            img[2, :, :] = tmp_img

        return img


class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, gray=2):
        self.gray = gray

    def __call__(self, img):

        idx = random.randint(0, self.gray)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
            img[0, :, :] = tmp_img
            img[1, :, :] = tmp_img
            img[2, :, :] = tmp_img
        return img


# class RandomErasing(object):
#     """ Randomly selects a rectangle region in an image and erases its pixels.
#         'Random Erasing Data Augmentation' by Zhong et al.
#         See https://arxiv.org/pdf/1708.04896.pdf
#     Args:
#          probability: The probability that the Random Erasing operation will be performed.
#          sl: Minimum proportion of erased area against input image.
#          sh: Maximum proportion of erased area against input image.
#          r1: Minimum aspect ratio of erased area.
#          mean: Erasing value.
#     """
#
#     def __init__(self, probability=0.5, sl=0.2, sh=0.8, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
#         self.probability = probability
#         self.mean = mean
#         self.sl = sl
#         self.sh = sh
#         self.r1 = r1
#
#     def __call__(self, img):
#
#         if random.uniform(0, 1) > self.probability:
#             return img
#
#         for attempt in range(100):
#             area = img.size()[1] * img.size()[2]
#
#             target_area = random.uniform(self.sl, self.sh) * area
#             aspect_ratio = random.uniform(self.r1, 1 / self.r1)
#
#             h = int(round(math.sqrt(target_area * aspect_ratio)))
#             w = int(round(math.sqrt(target_area / aspect_ratio)))
#
#             if w < img.size()[2] and h < img.size()[1]:
#                 x1 = random.randint(0, img.size()[1] - h)
#                 y1 = random.randint(0, img.size()[2] - w)
#                 if img.size()[0] == 3:
#                     img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
#                     img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
#                     img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
#                 else:
#                     img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
#                 return img
#
#         return img


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class mixing_erasing(object):
    def __init__(self,
                 probability=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=(0.4914, 0.4822, 0.4465),
                 mode='pixel',
                 type='normal',
                 mixing_coeff=[1.0, 1.0]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.rand_color = False
        self.per_pixel = False
        self.mode = mode
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal

        self.type = type
        self.mixing_coeff = mixing_coeff

        self.to_grayscale = transforms.Grayscale(num_output_channels=3)

    def __call__(self, img):
        area = img.size()[1] * img.size()[2]

        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, 1 / self.r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.size()[2] and h < img.size()[1]:
            # Application zone
            x1 = random.randint(0, img.size()[1] - h)
            y1 = random.randint(0, img.size()[2] - w)
            if self.type == 'normal':
                m = 1.0
            else:  # soft - soft_IR - self
                m = np.float32(
                    np.random.beta(self.mixing_coeff[0],
                                   self.mixing_coeff[1]))

            if self.type == "soft_RGB":
                img[:, x1:x1 + h, y1:y1 + w] = (1 - m) * img[:, x1:x1 + h, y1:y1 + w] + m * _get_pixels(self.per_pixel,
                                                                                                        self.rand_color,
                                                                                                        (img.size()[0], h, w),
                                                                                                        dtype=img.dtype)
            elif self.type == "soft_IR":
                img[:, x1:x1 + h, y1:y1 + w] = self.to_grayscale(
                    (1 - m) * img[:, x1:x1 + h, y1:y1 + w] + m * _get_pixels(
                        self.per_pixel, self.rand_color, (img.size()[0], h, w), dtype=img.dtype))
        return img


# def rgb2gray(rgb):
#     return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32):
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype)
