#!/usr/bin/python
import cv2
import numpy as np
from skimage import io, draw
from itertools import product 

#np.random.seed(21)
np.random.seed(332)
global count
count = 0
idxs = [d for d in np.random.multivariate_normal([2,2], [[10,0],[0,10]], 100000) if np.all(d < 5) and np.all(d > 0)]
idxs = np.uint8(np.array(idxs))
idxsn = [d for d in np.random.multivariate_normal([3,3], [[10,10],[0,10]], 100000) if np.all(d < 5) and np.all(d > 0)]
idxsn = np.uint8(np.array(idxsn))
idxs3 = [d for d in np.random.multivariate_normal([3,3], [[10,10],[0,10]], 100000) if np.all(d < 5) and np.all(d > 0)]
idxs3 = np.uint8(np.array(idxs3))

def create_pos(input_size, noise=False):
    global count
    img = np.zeros(input_size)
    coord = [2,2,2,5]
    img[draw.line(*coord)] = 255.
    coord = [2,5,5,5]
    img[draw.line(*coord)] = 255.
    coord = [5,5,5,2]
    img[draw.line(*coord)] = 255.
    coord = [5,2,2,2]
    img[draw.line(*coord)] = 255.
    if noise:
        a = np.zeros([5,5])
        b = np.zeros([5,5])
        for m, n in product(range(5), range(5)):
            a[m,n] = np.sum(idxs[count:count+10] == [m,n])*5.
            b[m,n] = np.sum(idxsn[count:count+10] == [m,n])*5.
        count+=10
        img[3:8,3:8]-=a
        img[1:6,2:7]+=b
    return img[:,:,None]

def create_neg(input_size, noise=False):
    global count
    img = np.zeros(input_size)
    coord = [2,2,2,5]
    img[draw.line(*coord)] = 255.
    coord = [2,5,5,5]
    img[draw.line(*coord)] = 255.
    coord = [5,5,5,2]
    img[draw.line(*coord)] = 255.
    if noise:
        a = np.zeros([5,5])
        b = np.zeros([5,5])
        for m, n in product(range(5), range(5)):
            a[m,n] = np.sum(idxsn[count:count+10] == [m,n])*5.
            b[m,n] = np.sum(idxs3[count:count+10] == [m,n])*5.
        count+=10
        img[3:8,3:8]-=a
        img[3:8,0:5]+=b
    return img[:,:,None]

a = np.zeros([5,5])
a = np.zeros([5,5])
for i, j in product(range(5), range(5)):
    a[i,j] = np.sum(idxs[count:count+10] == [i,j])*5
count+=10
print a
for i, j in product(range(5), range(5)):
    a[i,j] = np.sum(idxs[count:count+10] == [i,j])*5
count+=10
print a

def create_batch(batch_size, noise=False, input_size=(8, 8)):
    global count
    x = np.zeros([batch_size, input_size[0],input_size[1], 1])

    y = np.zeros([batch_size, 1, 1, 2])
    rands = np.random.rand(batch_size)
    for i in range(batch_size):
        if rands[i] > 0.5:
            x[i,:,:,:] = create_pos(input_size,noise=noise)
            y[i,0,0,:] = [1,0]
            """
            noise1 = np.random.normal(scale=20,size=[5,5])
            x[i,3:8,3:8,0]-=noise1
            """
        else:
            x[i,:,:,:] = create_neg(input_size,noise=noise)
            y[i,0,0,:] = [0,1]
            """
            noise2 = np.random.normal(scale=20,size=[5,5])
            x[i,3:8,3:8,0]-=noise2
            """
    return x, y



