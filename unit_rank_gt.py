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
modify=[]
for i in range(1,4):
    modify.append('conv%d'%i)

logits, net, activations, modifys = sc(inputs, modify=modify)
print modifys
modifyv = {}
for i in range(1,4):
    name = 'conv%d'%i
    print name
    modifyv[name] = np.ones(activations[name].shape)
with tf.Session() as sess:    
    saver = tf.train.Saver()
    saver.restore(sess,'ckpts2/39900.ckpt')
    fd = {modifys['conv%d'%i]:modifyv['conv%d'%i] for i in range(1,4)}
    fdpos = {inputs:pos}
    fdpos.update(fd)
    fdneg = {inputs:neg}
    fdneg.update(fd)
    posref = sess.run([net], feed_dict=fdpos)[0][0,0,0,0]
    negref = sess.run([net], feed_dict=fdneg)[0][0,0,0,1]
    #posref = sess.run([logits], feed_dict=fdpos)[0][0,0,0,0]
    #negref = sess.run([logits], feed_dict=fdneg)[0][0,0,0,1]
    print posref, negref
    negchs = {}
    poschs = {}
    posneg = {}
    for i in range(1,4):
        name='conv%d'%i
        _, height, width, channels = activations[name].shape
        zeroact = np.zeros([batch_size,height,width])
        for channel in range(channels):
            modifyvcopy = modifyv[name].copy()
            modifyvcopy[:,:,:,channel] = zeroact
            fd = {modifys['conv%d'%i]:modifyv['conv%d'%i] for i in range(1,4)}
            fd[modifys[name]] = modifyvcopy
            fdpos = {inputs:pos}
            fdpos.update(fd)
            fdneg = {inputs:neg}
            fdneg.update(fd)
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


