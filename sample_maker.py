#!/usr/bin/python
import cv2
import numpy as np
from skimage import io, draw

def create_pos(input_size):
    img = np.zeros(input_size)
    coord = [2,2,2,5]
    img[draw.line(*coord)] = 255
    coord = [2,5,5,5]
    img[draw.line(*coord)] = 255
    coord = [5,5,5,2]
    img[draw.line(*coord)] = 255
    coord = [5,2,2,2]
    img[draw.line(*coord)] = 255
    return img[:,:,None]

def create_neg(input_size):
    img = np.zeros(input_size)
    coord = [2,2,2,5]
    img[draw.line(*coord)] = 255
    coord = [2,5,5,5]
    img[draw.line(*coord)] = 255
    coord = [5,5,5,2]
    img[draw.line(*coord)] = 255
    return img[:,:,None]

def create_batch(batch_size, input_size=(8, 8)):
    x = np.zeros([batch_size, input_size[0],input_size[1], 1])
    

    y = np.zeros([batch_size, 1, 1, 2])
    rands = np.random.rand(batch_size)
    for i in range(batch_size):
        if rands[i] > 0.5:
            x[i,:,:,:] = create_pos(input_size)
            y[i,0,0,:] = [1,0]
        else:
            x[i,:,:,:] = create_neg(input_size)
            y[i,0,0,:] = [0,1]
    return x, y





input_size = (8, 8)
img = np.zeros(input_size)
coord = [2,2,2,5]
img[draw.line(*coord)] = 255
coord = [2,5,5,5]
img[draw.line(*coord)] = 255
coord = [5,5,5,2]
img[draw.line(*coord)] = 255
coord = [5,2,2,2]
img[draw.line(*coord)] = 255
#x = [4,7,7,4]
#y = [4,4,7,7]

#img[draw.polygon_perimeter(x, y, shape=img.shape, clip=True)] = 255
#img[draw.polygon_perimeter(x, y, shape=img.shape, clip=False)] = 255
io.imsave('sample.png',np.uint8(img))
