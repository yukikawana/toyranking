# coding: utf-8
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import tensorflow.contrib.slim as slim
import skimage.io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys, os
from collections import OrderedDict
from itertools import product
#from sc_freeze import sc
from sc_manual import sc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sample_maker import create_pos, create_neg 

# Create mini-batch for demo


input_shape = (8,8)
batch_size = 1
ch = 3


gradients = OrderedDict()
activations = OrderedDict()

inputs = tf.placeholder(tf.float32, shape=(batch_size, input_shape[0], input_shape[1], 1)) 
logits, net, activations = sc(inputs)
with tf.Session() as sess:    
    saver = tf.train.Saver()
    #saver.restore(sess,'ckpts_manual_noise/1.ckpt')
    #saver.restore(sess,'ckpts_manual_noise_0mean/1.ckpt')
    saver.restore(sess,'ckpts_manual_noise_gauss2/1.ckpt')
    for tv in tf.trainable_variables():
        tvv = sess.run(tv)
        if tv.name.find('weights') > -1:
            _, _, bchs, tchs = tvv.shape
            for bch, tch in product(range(bchs), range(tchs)):
                print tv.name, bch, tch
                print tvv[:,:,bch, tch]
        if tv.name.find('biases') > -1:
            tchs = tvv.shape[0]
            for tch in range(tchs):
                print tv.name, tch
                print tvv[tch]
            """
            if tv.name.find('biases')  > -1 or tv.name.find('scale') > -1:
                print tv.name, sess.run(tv)
            """
