# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:32:21 2021

@author: Eduin Hernandez
"""
import numpy as np
from skimage.util import view_as_windows as viewW


def im2col_2d(image, kernel_tuple):
    return viewW(image, (kernel_tuple[0], kernel_tuple[1])).reshape(-1,kernel_tuple[0]*kernel_tuple[1]).T

def im2col_3d(image, kernel_tuple):
    return viewW(image, (kernel_tuple[0], kernel_tuple[1], kernel_tuple[2])).reshape(-1, kernel_tuple[0]*kernel_tuple[1], kernel_tuple[2]).transpose(0,2,1)

def im2col_4d(image, kernel_tuple):
    return viewW(image, (image.shape[0], kernel_tuple[0], kernel_tuple[1], kernel_tuple[2])).reshape(-1,image.shape[0], kernel_tuple[0]*kernel_tuple[1], kernel_tuple[2]).transpose(1,0,3,2)

def conv2d_foward(image, kernel, result_shape): #4d Image and 4d Kernel
    h, w, d, f = kernel.shape
    num, row, col, depth = image.shape
    result = np.zeros(result_shape)
    
    for i0 in range(row - h + 1):
        for i1 in range(col - w + 1):
                result[:, i0, i1] = np.einsum('mijk, ijkl -> ml', image[:, i0:(i0 + h), i1:(i1 + w)], kernel)
    return result

def conv2d_weights(image, output, filter_shape): #4d Image and 4d Output Loss
    pass

