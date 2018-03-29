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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sample_maker import create_pos, create_neg
#from sc import sc
from sc_freeze import sc


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
    #saver.restore(sess,'ckpts5/39900.ckpt')
    saver.restore(sess,'ckpts_freeze3/9900.ckpt')
    fd = {modifys['conv%d'%i]:modifyv['conv%d'%i] for i in range(1,4)}
    fdpos = {inputs:pos}
    fdpos.update(fd)
    fdneg = {inputs:neg}
    fdneg.update(fd)
    posref = sess.run([activations], feed_dict=fdpos)[0]
    negref = sess.run([activations], feed_dict=fdneg)[0]
    #posref = sess.run([logits], feed_dict=fdpos)[0][0,0,0,0]
    #negref = sess.run([logits], feed_dict=fdneg)[0][0,0,0,1]
    negchs = OrderedDict()
    poschs = OrderedDict()
    posneg = OrderedDict()
    for i in reversed(range(2,5)):
        top='conv%d'%i
        bottom='conv%d'%(i-1)
        _, theight, twidth, tchannels = activations[top].shape
        _, bheight, bwidth, bchannels = activations[bottom].shape
        zeroact = np.zeros([batch_size,bheight,bwidth])
        for tchannel, bchannel in product(range(tchannels), range(bchannels)):
            modifyvcopy = modifyv[bottom].copy()
            modifyvcopy[:,:,:,bchannel] = zeroact
            fd = {modifys['conv%d'%i]:modifyv['conv%d'%i] for i in range(1,4)}
            fd[modifys[bottom]] = modifyvcopy
            fdpos = {inputs:pos}
            fdpos.update(fd)
            fdneg = {inputs:neg}
            fdneg.update(fd)
            posch = sess.run([activations[top]], feed_dict=fdpos)[0][0,:,:,tchannel].max()
            negch = sess.run([activations[top]], feed_dict=fdneg)[0][0,:,:,tchannel].max()
            #posch = sess.run([logits], feed_dict=fdpos)[0][0,0,0,0]
            #negch = sess.run([logits], feed_dict=fdneg)[0][0,0,0,1]
            posg = (posch - posref[top][0,:,:,tchannel].max())
            negg = (negch - negref[top][0,:,:,tchannel].max())
            poschs[(top, tchannel, bottom, bchannel)]=posg
            negchs[(top, tchannel, bottom, bchannel)]=negg
            posneg[(top, tchannel, bottom, bchannel)]=posg - negg
            print top, tchannel,bottom, bchannel, posg, negg, posg + negg
    """
    for k, v in sorted(poschs.items(), key=lambda x:x[1])[::-1]:
        print 'pos',k,v
    print '################'
    for k, v in sorted(negchs.items(), key=lambda x:x[1])[::-1]:
        print 'neg',k,v
    print '################'
    for k, v in sorted(posneg.items(), key=lambda x:x[1])[::-1]:
        print 'pnn',k,v
    """


