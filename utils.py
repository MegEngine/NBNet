#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-22 22:07:08

import math
import torch
import megengine.functional as F
from skimage import img_as_ubyte
import numpy as np
import cv2


def get_gausskernel(p, chn=3):
    '''
    Build a 2-dimensional Gaussian filter with size p
    '''
    x = cv2.getGaussianKernel(p, sigma=-1)   # p x 1
    y = np.matmul(x, x.T)[np.newaxis, np.newaxis,]  # 1x 1 x p x p
    out = np.tile(y, (chn, 1, 1, 1)) # chn x 1 x p x p

    return torch.from_numpy(out).type(torch.float32)

def gaussblur(x, kernel, p=5, chn=3):
    x_pad = F.pad(x, pad=[int((p-1)/2),]*4, mode='reflect')
    y = F.conv2d(x_pad, kernel, padding=0, stride=1, groups=chn)

    return y


def calculate_psnr(im1, im2, border=0):
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    mse = F.mean((im1 - im2)**2)
    if mse == 0:
        return float('inf')
    return 10 * F.log(1.0 / mse) / F.log(10.)

def batch_PSNR(img, imclean, border=0):
    Img = img
    Iclean = imclean
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += calculate_psnr(Iclean[i,:,], Img[i,:,], border)
    return (PSNR/Img.shape[0])




def MixUp_AUG(rgb_gt, rgb_noisy):
    bs = rgb_gt.shape[0]
    indices = np.arange(bs)
    np.random.shuffle(indices)
    rgb_gt2 = rgb_gt[indices]
    rgb_noisy2 = rgb_noisy[indices]

    lam = np.random.beta(1.2, 1.2, (bs, 1, 1, 1))

    rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
    rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

    return rgb_gt, rgb_noisy

