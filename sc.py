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

def sc(inputs, input_shape=(8,8), batch_size=1, ch=3, modify=[]):
    activations = OrderedDict()
    padding = 'SAME'
    initializer = tf.truncated_normal_initializer(stddev=0.01)
    regularizer = slim.l2_regularizer(0.0005)
    modifys = {}

    name = 'conv1'
    net = slim.conv2d(inputs, ch, [3, 3],
    padding=padding,
    weights_initializer=initializer,
    weights_regularizer=regularizer,
    scope=name)
    activations[name] = net
    if name in modify:
        modifys[name]= tf.placeholder(tf.float32, shape=(batch_size, None, None, None)) 
        net = tf.multiply(net, modifys[name])

    net = slim.max_pool2d(net, [2, 2], scope='pool1')

    name = 'conv2'
    net = slim.conv2d(net, 2, [3, 3],
        padding=padding,
        weights_initializer=initializer,
        weights_regularizer=regularizer,
        scope=name)
    activations[name] = net
    if name in modify:
        modifys[name]= tf.placeholder(tf.float32, shape=(batch_size, None, None, None)) 
        net = tf.multiply(net, modifys[name])

    net = slim.max_pool2d(net, [2, 2], scope='pool2')

    name='conv3'
    net = slim.conv2d(net, 2, [3, 3],
        padding=padding,
        weights_initializer=initializer,
        weights_regularizer=regularizer,
        scope=name)
    activations[name] = net
    if name in modify:
        modifys[name]= tf.placeholder(tf.float32, shape=(batch_size, None, None, None)) 
        net = tf.multiply(net, modifys[name])

    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.conv2d(net, 2, [3, 3],
        padding=padding,
        weights_initializer=initializer,
        weights_regularizer=regularizer,
        activation_fn=None,
        scope='conv4')
    activations['conv4'] = net
    print net
    logits = slim.softmax(net) 
    if len(modify) != 0:
        return logits, net, activations, modifys
    else:
        return logits, net, activations
    
