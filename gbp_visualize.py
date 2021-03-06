
# coding: utf-8

# # GradCAM Visualization Demo with VGG16
# 

# In[1]:


# Replace vanila relu to guided relu to get guided backpropagation.
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from deepdreamlib import render_deepdream
import skimage.io
import numpy as np
from segnet import segnet
import utils
import cv2
import matplotlib.pyplot as plt
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

# Create mini-batch for demo


input_shape = (360, 480)

batch_img = cv2.resize(cv2.imread('car.png'), (input_shape[1], input_shape[0]))

batch_img = np.expand_dims(batch_img,0 )
batch_size = 1

topk = 0
name = 'conv3_3_D'
name = 'conv4_2'
name = 'conv5_3'
obj = 9

# Create tensorflow graph for evaluation
eval_graph = tf.Graph()
with eval_graph.as_default():
    #with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):

    images = tf.placeholder("float", [batch_size, input_shape[0], input_shape[1], 3])
    masks = tf.placeholder("float", [batch_size, input_shape[0], input_shape[1],12])

    logits, end_points = segnet(images)
    prob = tf.nn.softmax(logits)
    argmax = tf.argmax(logits, axis=3)

    cost = tf.reduce_sum(tf.multiply(logits, masks))
    grad_for_weighting = tf.gradients(cost, end_points[name])[0]
    pixelwise_weight = tf.abs(tf.multiply(grad_for_weighting, end_points[name]))

    layerwise_weights = tf.squeeze(tf.reduce_mean(pixelwise_weight, axis=(1,2)))

        
# Run tensorflow 

with tf.Session(graph=eval_graph) as sess:    
    saver = tf.train.Saver()
    saver.restore(sess,'weights/segnet_tf.ckpt')
    #sess.run(init)
    
    
    #compute mask for final output
    argmax_value = sess.run([argmax], feed_dict={images: batch_img})[0]
    npmasks = np.zeros([1,input_shape[0], input_shape[1], 12])
    npmask = np.zeros([1,input_shape[0], input_shape[1]])
    npmask[argmax_value==obj] = 1

    skimage.io.imsave('mask.png',np.uint8(npmask[0]*255))
    skimage.io.imsave('mas2k.png',np.uint8(argmax_value[0]/12.*255))
    #npmasks[:,:,:,obj] = npmask
    npmasks[0,180,160,obj] = 1
    pixelwise_weights_value = sess.run([pixelwise_weight], feed_dict={images: batch_img, masks: npmasks})[0][0]
    pixelwise_indices = np.array(np.unravel_index(np.argsort(pixelwise_weights_value, axis=None)[::-1], pixelwise_weights_value.shape)).transpose()
    pixelwise_importance = np.array(np.sort(pixelwise_weights_value, axis=None)[::-1])
    pixelwise_importance = (pixelwise_importance - np.min(pixelwise_importance))/(np.ptp(pixelwise_importance)+1e-7)
    layerwise_indices = np.array(np.argsort(np.mean(pixelwise_weights_value, axis=(0,1)))[::-1])
    channel_indices = pixelwise_indices[:, 2]
    channel_importance = []
    for ind in xrange(np.max(channel_indices)+1):
        channel_importance.append(np.sum(pixelwise_importance[channel_indices==ind]))
    new_layerwise_indices = np.argsort(channel_importance)[::-1]

    print pixelwise_indices[0:31,:]
    print layerwise_indices[0:11]
    print new_layerwise_indices[0:11]
    """
    weights_sorted = np.sort(weights_value)[::-1]
    percentage_pre = (weights_sorted - np.min(weights_sorted))/np.ptp(weights_sorted)
    percentage = percentage_pre/np.sum(percentage_pre)
#    plt.plot(percentage)
    #plt.plot(np.cumsum(percentage))
#    plt.savefig("savefig.png")
    """
channel = pixelwise_indices[topk,2]
eval_graph = tf.Graph()
with eval_graph.as_default():
    inputs = tf.placeholder("float", shape=(1, input_shape[0],input_shape[1], 3))
    logits, end_points = segnet(inputs)
    grad = tf.gradients(tf.reduce_mean(end_points[name][:,:,:,channel]), inputs)[0]

    sess = tf.Session(graph=eval_graph)
    saver = tf.train.Saver()
    saver.restore(sess,'weights/segnet_tf.ckpt')
#imgs = render_deepdream(end_points['relu5_3'][:,:,:,386],inputs,  img_noise, sess)
img = render_deepdream(sess, inputs, grad, iter_n=30, octave_n=4)
skimage.io.imsave('0.png',img)
print channel, name
