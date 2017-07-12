#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   BackPropagation 

File Name:  maxpool.py

"""

__author__ = 'Welkin'
__date__ = '2017/7/12 01:02'

from PIL import Image
from scipy.misc import imread, imsave
import numpy as np
# import scipy
import matplotlib.pyplot as plt

def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
    #     im.show()
    width, height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data, dtype = 'float') / 255.0
    # new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data, (height, width))
    return new_data

#     new_im = Image.fromarray(new_data)
#     # 显示图片
#     new_im.show()

def ImageToMatrix2(filename):
    img = imread(filename)
    print(img.shape)
    # (600, 1000, 3)
    return img

def MatrixToImage(data):
    data = data * 255
    print(data.shape)
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


# 2*2 池化
def maxpool(data):
    height, weight = data.shape
    print(height // 2, weight // 2)
    new_data = np.zeros([height // 2, weight // 2])
    for i in range(height // 2):
        for j in range(weight // 2):
            new_data[i, j] = np.average(data[2 * i: 2 * i + 2, 2 * j: 2 * j + 2])
    return new_data

def resize(data):
    height, weight = data.shape
    print(height * 2, weight * 2)
    new_data = np.zeros([height * 2, weight * 2])
    for i in range(height):
        for j in range(weight):
            new_data[2 * i: 2 * i + 2, 2 * j: 2 * j + 2] = data[i, j]
    return new_data

def save_maxpool_3d(filename):

    data = ImageToMatrix2(filename)
    data = data / 255.0
    new_data = np.zeros([data.shape[0] // 2, data.shape[1] // 2, data.shape[2]])
    for i in [0, 1, 2]:
        data2 = data[:, :, i]
        new_data2 = maxpool(data2)
        # new_im = MatrixToImage(new_data2)
        # new_im.show()
        # new_im.save('alpha_' + str(i) + '.jpg')
        new_data[:, :, i] = new_data2
    imsave('new_im_average.jpg', new_data)
    return new_data

def resize_3d(filename):
    data = ImageToMatrix2(filename)
    data = data / 255.0
    resize_data = np.zeros([data.shape[0] * 2, data.shape[1] * 2, data.shape[2]])
    for i in [0, 1, 2]:
        data2 = data[:, :, i]
        resize_data2 = resize(data2)
        resize_data[:, :, i] = resize_data2

    imsave('new_im_max_re.jpg', resize_data)
    return resize_data


if __name__ == '__main__':
    # filename = 'IMG_5930.JPG'
    # new_data = save_maxpool_3d(filename)

    filename = "new_im_max.jpg"
    resize_data = resize_3d(filename)



    # print(data)
    # new_im = MatrixToImage(data)
    # plt.imshow(data, cmap = plt.cm.gray, interpolation = 'nearest')
    # new_im.show()
    # new_im.save('lena_1.bmp')


