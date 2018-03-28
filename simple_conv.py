#!/usr/bin/python
import cv2
import numpy as np
from skimage import io, draw
import tensorflow.contrib.slim as slim
import tensorflow as tf
from sample_maker import create_batch

padding = 'SAME'
initializer = tf.truncated_normal_initializer(stddev=0.01)
regularizer = slim.l2_regularizer(0.0005)
#regularizer = slim.l1_regularizer(0.000005)
ch = 3
batch_size = 50 

sess = tf.InteractiveSession()

input_shape=(8,8)
inputs = tf.placeholder(tf.float32, shape=(batch_size, input_shape[0], input_shape[1], 1))
labels = tf.placeholder(tf.float32, shape=(batch_size, 1, 1, 2))

net = slim.conv2d(inputs, ch, [3, 3],
    padding=padding,
    weights_initializer=initializer,
    weights_regularizer=regularizer,
    scope='conv1')
print net

net = slim.max_pool2d(net, [2, 2], scope='pool1')
print net
net = slim.conv2d(net, 2, [3, 3],
    padding=padding,
    weights_initializer=initializer,
    weights_regularizer=regularizer,
    scope='conv2')
print net
net = slim.max_pool2d(net, [2, 2], scope='pool2')
print net
net = slim.conv2d(net, 2, [3, 3],
    padding=padding,
    weights_initializer=initializer,
    weights_regularizer=regularizer,
    scope='conv3')
print net
net = slim.max_pool2d(net, [2, 2], scope='pool3')
print net
net = slim.conv2d(net, 2, [3, 3],
    padding=padding,
    weights_initializer=initializer,
    weights_regularizer=regularizer,
    activation_fn=None,
    scope='conv4')
print net
logits = slim.softmax(net)
loss = tf.losses.softmax_cross_entropy(labels,net)
tf.losses.add_loss(loss)
total_loss = tf.losses.get_total_loss()
tf.summary.scalar('losses/total_loss', total_loss)
#train_step = tf.train.AdamOptimizer(1e-5).minimize(total_loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
for i in xrange(40000):
    x, y= create_batch(50)
    #normalize and centralize the batch
    #x = x/255
    if i%100 == 0:
        train_accuracy = loss.eval(feed_dict={inputs:x, labels:y})
        loval = logits.eval(feed_dict={inputs:x, labels:y})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        saver.save(sess, 'ckpts5/%d.ckpt'%(i))
        
    train_step.run(feed_dict={inputs:x, labels:y})
"""
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
train_tensor = slim.learning.create_train_op(total_loss, optimizer)
slim.learning.train(train_tensor, train_log_dir)
"""
