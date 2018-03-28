#!/usr/bin/python
import cv2
import numpy as np
from skimage import io, draw

input_size = (16, 16)
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
