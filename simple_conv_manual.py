#!/usr/bin/python
import cv2
import numpy as np
from skimage import io, draw
import tensorflow.contrib.slim as slim
import tensorflow as tf
from sample_maker import create_batch
from sc_manual import sc
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 50 
input_shape=(8,8)

inputs = tf.placeholder(tf.float32, shape=(batch_size, input_shape[0], input_shape[1], 1))
logits, net, activations= sc(inputs)


sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
#saver.restore(sess,'ckpts_freeze/20000.ckpt')

for tv in tf.trainable_variables():
    #weight is h, w, bch, tch
    w = np.zeros(tv.get_shape().as_list())
    if tv.name.find('conv1')>-1:
        if tv.name.find('weights')>-1:
            w = np.zeros(tv.get_shape().as_list())
            w[:,:,0,0] = [[1,1,1],
                          [1,0,0],
                          [1,0,0]]

            w[:,:,0,1] = [[1,0,0],
                          [1,0,0],
                          [1,1,1]]

            w[:,:,0,2] = [[0,0,0],
                          [1,1,1],
                          [0,0,0]]
            w/=255.
            sess.run(tv.assign(w))
        if tv.name.find('biases')>-1:
            b = np.zeros(tv.get_shape().as_list())
            b[0] = -4
            b[1] = -4
            b[2] = -2
            sess.run(tv.assign(b))

    elif tv.name.find('conv2')>-1:
        if tv.name.find('weights')>-1:
            w = np.zeros(tv.get_shape().as_list())
            w[:,:,0,0] = [[0,0,0],
                          [0,1,0],
                          [0,0,0]]

            w[:,:,1,0] = [[0,0,0],
                          [0,1,0],
                          [0,0,0]]

            w[:,:,2,0] = [[0,0,0],
                          [0,1,1],
                          [0,1,1]]

            w[:,:,0,1] = [[0,0,0],
                          [0,-1,0],
                          [0,0,0]]

            w[:,:,1,1] = [[0,0,0],
                          [0,0,0],
                          [0,-1,0]]

            w[:,:,2,1] = [[0,0,0],
                          [0,1,1],
                          [0,1,1]]
            sess.run(tv.assign(w))
        if tv.name.find('biases')>-1:
            b = np.zeros(tv.get_shape().as_list())
            b[0] = 0
            b[1] = -3
            sess.run(tv.assign(b))

    elif tv.name.find('conv3')>-1:
        if tv.name.find('weights')>-1:
            w = np.zeros(tv.get_shape().as_list())
            w[:,:,0,0] = [[0,0,0],
                          [0,1,0],
                          [0,1,0]]

            w[:,:,1,0] = [[0,0,0],
                          [0,-1,0],
                          [0,0,0]]

            w[:,:,0,1] = [[0,0,0],
                          [0,0,0],
                          [0,0,0]]

            w[:,:,1,1] = [[0,0,0],
                          [0,1,0],
                          [0,0,0]]
            sess.run(tv.assign(w))
        if tv.name.find('biases')>-1:
            b = np.zeros(tv.get_shape().as_list())
            b[0] = -7
            b[1] = 0
            sess.run(tv.assign(b))

#saver.save(sess,'freeze.ckpt')
saver.save(sess, 'ckpts_manual/%d.ckpt'%(1))
#saver.restore(sess,'freeze.ckpt')
