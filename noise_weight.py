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

from scipy import fftpack, signal

input_shape = (8,8)
batch_size = 1


neg = create_neg(input_shape)[None,:,:,:]
pos= create_pos(input_shape)[None,:,:,:]

np.random.seed(21)
noise = np.random.normal(scale=10,size=[5,5])
noise2 = np.random.normal(scale=10,size=[5,5])
posn = pos.copy()
negn = neg.copy()
posn[0,3:8,3:8,0]-=noise
negn[0,3:8,3:8,0]-=noise2



inputs = tf.placeholder(tf.float32, shape=(batch_size, input_shape[0], input_shape[1], 1)) 

logits, net, activations = sc(inputs)
with tf.Session() as sess:    
    saver = tf.train.Saver()
    #saver.restore(sess,'ckpts5/39900.ckpt')
    saver.restore(sess,'ckpts_manual/1.ckpt')

    vs = tf.trainable_variables()
    wv = vs[0]
    print wv.name
    w = sess.run(wv)
    print w[:,:,0,0]
    print w[:,:,0,1]
    print w[:,:,0,2]
    kernel_ft = fftpack.fft2(w[:,:,0,0], shape=pos.shape[1:3], axes=(0, 1))
    img_ft = fftpack.fft2(pos[0,:,:,0], axes=(0, 1))
    img2_ft = kernel_ft * img_ft
    img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real
    print np.uint8(img2)
    img3 = signal.fftconvolve(pos[0,:, :, 0],w[:,:,0,0], mode='same')
    print np.uint8(img3)

    fdpos = {inputs:pos}
    fdneg = {inputs:neg}
    fdposn = {inputs:posn}
    fdnegn = {inputs:negn}
    posref = sess.run([activations], feed_dict=fdpos)[0]
    negref = sess.run([activations], feed_dict=fdneg)[0]
    posnref = sess.run([activations], feed_dict=fdposn)[0]
    negnref = sess.run([activations], feed_dict=fdnegn)[0]



    print posref['conv3'][0,:,:,0]
    print ""
    print posnref['conv3'][0,:,:,0]
    print ""
    print negref['conv3'][0,:,:,1]
    print ""
    print negnref['conv3'][0,:,:,1]
    posres = sess.run([net], feed_dict=fdposn)[0][0,0,0,:]
    negres = sess.run([net], feed_dict=fdnegn)[0][0,0,0,:]
    print posres, negres
    negchs = OrderedDict()
    poschs = OrderedDict()
    posneg = OrderedDict()
    for i in reversed(range(2,4)):
        top='conv%d'%i
        bottom='conv%d'%(i-1)
        _, theight, twidth, tchannels = activations[top].shape
        _, bheight, bwidth, bchannels = activations[bottom].shape
        zeroact = np.zeros([batch_size,bheight,bwidth])
        for tchannel, bchannel in product(range(tchannels), range(bchannels)):
            fdpos = {inputs:pos}
            fdneg = {inputs:neg}
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


