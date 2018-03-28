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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sample_maker import create_pos, create_neg
from sc import sc


input_shape = (8,8)
batch_size = 1
ch = 3


neg = create_neg(input_shape)[None,:,:,:]
pos= create_pos(input_shape)[None,:,:,:]

inputs = tf.placeholder(tf.float32, shape=(batch_size, input_shape[0], input_shape[1], 1)) 

logits, net, activations = sc(inputs)
with tf.Session() as sess:    
    saver = tf.train.Saver()
    saver.restore(sess,'ckpts2/39900.ckpt')
    negchs = {}
    poschs = {}
    posneg = {}
    for i in range(1,4):
        name='conv%d'%i
        _, height, width, channels = activations[name].shape
        for channel in range(channels):
            max_idx = np.argmax(activations[name][0,:,:,channel])
            y, x = np.unravel_index(max_idx, [height, width])
            npmask = np.zeros(shape)
            npmask[0,y,x,channel] = 1.
            grad = sess.run([gradients[name]],feed_dict={inputs:img, mask:npmask})[0]
            fdpos = {inputs:pos}
            fdneg = {inputs:neg}
            posch = sess.run([net], feed_dict=fdpos)[0][0,0,0,0]
            negch = sess.run([net], feed_dict=fdneg)[0][0,0,0,1]
            #posch = sess.run([logits], feed_dict=fdpos)[0][0,0,0,0]
            #negch = sess.run([logits], feed_dict=fdneg)[0][0,0,0,1]
            posg = (posref-posch)
            negg = (negref-negch)
            poschs[(name, channel)]=posg
            negchs[(name, channel)]=negg
            posneg[(name, channel)]=posg - negg
            print name, channel, posch, negch
    for k, v in sorted(poschs.items(), key=lambda x:x[1])[::-1]:
        print 'pos',k,v
    print '################'
    for k, v in sorted(negchs.items(), key=lambda x:x[1])[::-1]:
        print 'neg',k,v
    print '################'
    for k, v in sorted(posneg.items(), key=lambda x:x[1])[::-1]:
        print 'pnn',k,v


