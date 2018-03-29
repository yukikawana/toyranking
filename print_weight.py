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

inputs = tf.placeholder(tf.float32, shape=(batch_size, input_shape[0], input_shape[1], 1)) 
logits, net, activations = sc(inputs)
with tf.Session() as sess:    
    saver = tf.train.Saver()
    saver.restore(sess,'ckpts_freeze_ng/9900.ckpt')
    for tv in tf.trainable_variables():
        if tv.name.find('biases')  > -1 or tv.name.find('scale') > -1:
            print tv.name, sess.run(tv)
