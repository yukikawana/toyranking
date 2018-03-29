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
from sc_freeze import sc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sample_maker import create_pos, create_neg 
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

# Create mini-batch for demo


input_shape = (8,8)
batch_size = 1
ch = 3


gradients = OrderedDict()
activations = OrderedDict()

eval_graph = tf.Graph()
with eval_graph.as_default():
    with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
    
        inputs = tf.placeholder(tf.float32, shape=(batch_size, input_shape[0], input_shape[1], 1)) 
        mask = tf.placeholder(tf.float32, shape=(batch_size, None, None, None)) 

        logits, net, activations = sc(inputs)
        for name in activations:
            masked_act = tf.multiply(activations[name],mask)
            cost = tf.reduce_sum(masked_act)
            gradients[name]= tf.gradients(cost, inputs)[0]


with tf.Session(graph=eval_graph) as sess:    
    saver = tf.train.Saver()
    saver.restore(sess,'ckpts_freeze_ng/9900.ckpt')
    for tv in tf.trainable_variables():
        print tv.name, sess.run(tv)
    neg = create_neg(input_shape)[None,:,:,:]
    neg-=128
    actneg = sess.run([activations], feed_dict={inputs: neg})[0]
    pos= create_pos(input_shape)[None,:,:,:]
    pos-=128
    actpos = sess.run([activations], feed_dict={inputs: pos})[0]
    posref = sess.run([logits], feed_dict={inputs: pos})[0][0,0,0,:]
    negref = sess.run([logits], feed_dict={inputs: neg})[0][0,0,0,:]
    print posref, negref

    for name in actpos:
        shape = actpos[name].shape
        for channel in range(shape[3]):
            print "pos neg",name, channel, actpos[name][:,:,:,channel].max() , actneg[name][:,:,:,channel].max()
            if actpos[name][:,:,:,channel].max() > actneg[name][:,:,:,channel].max():
                act = actpos
                img = pos
            else:
                act = actneg
                img = neg
            max_idx = np.argmax(act[name][0,:,:,channel])
            y, x = np.unravel_index(max_idx, shape[1:3])
            npmask = np.zeros(shape)
            npmask[0,y,x,channel] = 1.
            grad = sess.run([gradients[name]],feed_dict={inputs:img, mask:npmask})[0]
            nonzero = grad[0,:,:,0].nonzero()
            try:
                maxy = nonzero[0].max()
                miny = nonzero[0].min()
                maxx = nonzero[1].max()
                minx = nonzero[1].min()
            except:
                continue
            if maxy-miny == 0 or maxx - minx == 0:
                continue
            img+=128
            cropped = np.uint8(img[0,miny:maxy+1,minx:maxx+1, 0])
            gcropped = grad[0,miny:maxy+1,minx:maxx+1, 0]
            print 'imgsize = ',maxy-miny+1, maxx-minx+1,cropped.shape
            cropped = cv2.resize(cropped,(input_shape[0]*10, input_shape[1]*10), interpolation=cv2.INTER_NEAREST)
            gcropped = (gcropped-gcropped.min())/(np.ptp(gcropped)+1e-12)
            gcropped = np.uint8(gcropped*255).clip(0,255) 
            gcropped = cv2.resize(gcropped,(input_shape[0]*10, input_shape[1]*10), interpolation=cv2.INTER_NEAREST)
            gcropped = cv2.applyColorMap(gcropped, cv2.COLORMAP_JET)

            skimage.io.imsave('vis_freeze_ng/%s_%d.png'%(name,channel),cropped)
            skimage.io.imsave('vis_freeze_ng/%s_%d_g.png'%(name,channel),gcropped)
