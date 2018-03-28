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
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

# Create mini-batch for demo


input_shape = (8,8)
batch_size = 1
ch = 3

padding = 'SAME'
initializer = tf.truncated_normal_initializer(stddev=0.01)
regularizer = slim.l2_regularizer(0.0005)

gradients = OrderedDict()
activations = OrderedDict()

eval_graph = tf.Graph()
with eval_graph.as_default():
    with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
    
        inputs = tf.placeholder(tf.float32, shape=(batch_size, input_shape[0], input_shape[1], 1)) 
        mask = tf.placeholder(tf.float32, shape=(batch_size, None, None, None)) 

        net = slim.conv2d(inputs, ch, [3, 3],
        padding=padding,
        weights_initializer=initializer,
        weights_regularizer=regularizer,
        scope='conv1')
        activations['conv1'] = net
        print net

        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        print net
        net = slim.conv2d(net, 2, [3, 3],
            padding=padding,
            weights_initializer=initializer,
            weights_regularizer=regularizer,
            scope='conv2')
        activations['conv2'] = net
        print net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        print net
        net = slim.conv2d(net, 2, [3, 3],
            padding=padding,
            weights_initializer=initializer,
            weights_regularizer=regularizer,
            scope='conv3')
        activations['conv3'] = net
        print net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        print net
        net = slim.conv2d(net, 2, [3, 3],
            padding=padding,
            weights_initializer=initializer,
            weights_regularizer=regularizer,
            activation_fn=None,
            scope='conv4')
        activations['conv4'] = net
        print net
        logits = slim.softmax(net) 
        for name in activations:
            masked_act = tf.multiply(activations[name],mask)
            cost = tf.reduce_sum(masked_act)
            gradients[name]= tf.gradients(cost, inputs)[0]


with tf.Session(graph=eval_graph) as sess:    
    saver = tf.train.Saver()
    saver.restore(sess,'ckpts4/39900.ckpt')
    neg = create_neg(input_shape)[None,:,:,:]
    actneg = sess.run([activations], feed_dict={inputs: neg})[0]
    pos= create_pos(input_shape)[None,:,:,:]
    actpos = sess.run([activations], feed_dict={inputs: pos})[0]

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
            maxy = nonzero[0].max()
            miny = nonzero[0].min()
            maxx = nonzero[1].max()
            minx = nonzero[1].min()
            if maxy-miny == 0 or maxx - minx == 0:
                continue
            cropped = np.uint8(img[0,miny:maxy+1,minx:maxx+1, 0])
            gcropped = grad[0,miny:maxy+1,minx:maxx+1, 0]
            print 'imgsize = ',maxy-miny+1, maxx-minx+1,cropped.shape
            cropped = cv2.resize(cropped,(input_shape[0]*10, input_shape[1]*10), interpolation=cv2.INTER_NEAREST)
            gcropped = (gcropped-gcropped.min())/(np.ptp(gcropped)+1e-12)
            gcropped = np.uint8(gcropped*255).clip(0,255) 
            gcropped = cv2.resize(gcropped,(input_shape[0]*10, input_shape[1]*10), interpolation=cv2.INTER_NEAREST)
            gcropped = cv2.applyColorMap(gcropped, cv2.COLORMAP_JET)

            skimage.io.imsave('vis4/%s_%d.png'%(name,channel),cropped)
            skimage.io.imsave('vis4/%s_%d_g.png'%(name,channel),gcropped)
