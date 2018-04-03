#!/usr/bin/python
import cv2
import numpy as np
from skimage import io, draw
import tensorflow.contrib.slim as slim
import tensorflow as tf
from sample_maker import create_batch, create_neg, create_pos
from sc_manual import sc
import os, sys
from itertools import product
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 50 
input_shape=(8,8)

inputs = tf.placeholder(tf.float32, shape=(batch_size, input_shape[0], input_shape[1], 1))
labels = tf.placeholder(tf.float32, shape=(batch_size, None, None, None))
logits, net, activations= sc(inputs)


sess = tf.InteractiveSession()
saver = tf.train.Saver()
#sess.run(tf.global_variables_initializer())
#saver.restore(sess,'ckpts_freeze/20000.ckpt')

neg = create_neg(input_shape)[:,:,:]
pos= create_pos(input_shape)[:,:,:]
poss = np.array([pos for _ in xrange(batch_size)])
negs = np.array([neg for _ in xrange(batch_size)])

trainingv = ['conv%d/weights:0'%j for j in xrange(1,4)]
trr = [tv for tv in tf.trainable_variables() if tv.name in trainingv]
labelss = {}
for j in range(1,4):
    labelss['conv%d'%j] = tf.placeholder(tf.float32, shape=activations['conv%d'%j].get_shape().as_list())
    loss = tf.losses.mean_squared_error(labelss['conv%d'%j],activations['conv%d'%j],weights=100./j**2)
    tf.losses.add_loss(loss)
total_loss = tf.losses.get_total_loss()
    #loss = tf.losses.softmax_cross_entropy(labels,activations['conv%d'%j],weights=10)
#train_step = tf.train.AdamOptimizer(1e-6).minimize(total_loss, var_list=trr)
train_step = tf.train.AdamOptimizer(1e-5).minimize(total_loss, var_list=trr)


sess.run(tf.global_variables_initializer())
saver.restore(sess,'ckpts_manual/1.ckpt')

tvs = tf.trainable_variables()
print sess.run(tvs[2])[:,:,0,1]
posref = sess.run([activations], feed_dict={inputs:poss})[0]
negref = sess.run([activations], feed_dict={inputs:negs})[0]
for i in xrange(800000):
    x, y= create_batch(batch_size, noise=True)
    feed_dict={inputs:x}
    for j in range(1,4):
        shape = posref['conv%d'%j].shape[1:]
        lb = np.zeros([batch_size, shape[0], shape[1], shape[2]])
        for k in range(batch_size):
            lb[k,:,:,:] = posref['conv%d'%j][0,:,:,:] if y[k,0,0,0] == 1 else negref['conv%d'%j][0,:,:,:]
            feed_dict.update({labelss['conv%d'%j]:lb})
    #normalize and centralize the batch
    #x = x/255
    if i%100 == 0:
        train_accuracy = total_loss.eval(feed_dict=feed_dict)
        print("step %d, training accuracy %g"%(i, train_accuracy))
        if train_accuracy == 0:
            break
        
    train_step.run(feed_dict=feed_dict)
tvs = tf.trainable_variables()
print sess.run(tvs[2])[:,:,0,1]


"""
print sess.run(tvs[0])[:,:,0,1]
assert(False)
"""

"""
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
train_tensor = slim.learning.create_train_op(total_loss, optimizer)
slim.learning.train(train_tensor, train_log_dir)
"""
saver.save(sess, 'ckpts_manual_noise_gauss2/1.ckpt')
