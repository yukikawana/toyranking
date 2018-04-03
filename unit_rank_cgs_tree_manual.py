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
from sc_manual import sc


input_shape = (8,8)
batch_size = 1


neg = create_neg(input_shape)[None,:,:,:]
pos= create_pos(input_shape)[None,:,:,:]

inputs = tf.placeholder(tf.float32, shape=(batch_size, input_shape[0], input_shape[1], 1)) 

logits, net, activations = sc(inputs)
with tf.Session() as sess:    
    saver = tf.train.Saver()
    #saver.restore(sess,'ckpts5/39900.ckpt')
    saver.restore(sess,'ckpts_manual/1.ckpt')
    fdpos = {inputs:pos}
    fdneg = {inputs:neg}
    posref = sess.run([activations], feed_dict=fdpos)[0]
    negref = sess.run([activations], feed_dict=fdneg)[0]
    negchs = OrderedDict()
    poschs = OrderedDict()
    posneg = OrderedDict()
    for i in reversed(range(2,4)):
        top='conv%d'%i
        bottom='conv%d'%(i-1)
        _, theight, twidth, tchannels = activations[top].shape
        _, bheight, bwidth, bchannels = activations[bottom].shape
        zeroact = np.zeros([batch_size,bheight,bwidth])
        #for tchannel, bchannel in product(range(tchannels), range(bchannels)):
        for tchannel in range(tchannels):
            px, py = np.unravel_index(sess.run([activations[top]], feed_dict=fdpos)[0][0,:,:,tchannel].argmax(), [theight, twidth])
            nx, ny = np.unravel_index(sess.run([activations[top]], feed_dict=fdneg)[0][0,:,:,tchannel].argmax(), [theight, twidth])
            for bchannel in range(bchannels):
                mask = np.zeros([batch_size, bheight, bwidth, bchannels])
                mask[0,:,:,bchannel] = 1
                grad = tf.gradients(activations[top][0,:,:,tchannel], activations[bottom])[0]
                pgrad = sess.run([grad[:,:,:,bchannel]], feed_dict=fdpos)[0]
                print pgrad
                ngrad = sess.run([grad], feed_dict=fdneg)[0]
                print bottom, bchannel, '->' , top, tchannel
                print pgrad[0,:,:,bchannel]
                print ''
                print ngrad[0,:,:,bchannel]
            #print top, tchannel,bottom, bchannel, posg, negg, posg + negg
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


