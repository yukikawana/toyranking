#!/usr/bin/python
import cv2
import numpy as np
from skimage import io, draw
import tensorflow.contrib.slim as slim
import tensorflow as tf
from sample_maker import create_batch
from sc_freeze import sc
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 50 
input_shape=(8,8)

inputs = tf.placeholder(tf.float32, shape=(batch_size, input_shape[0], input_shape[1], 1))
labels = tf.placeholder(tf.float32, shape=(batch_size, 1, 1, 2))
logits, net, activations= sc(inputs)


loss = tf.losses.softmax_cross_entropy(labels,net)
tf.losses.add_loss(loss)
total_loss = tf.losses.get_total_loss()
sess = tf.InteractiveSession()
saver = tf.train.Saver()
trainvs = [tv for tv in tf.trainable_variables() if tv.name.find('biases') > -1 or tv.name.find('scale') > -1]
print trainvs
train_step = tf.train.AdamOptimizer(1e-2).minimize(total_loss,var_list=trainvs)
sess.run(tf.global_variables_initializer())
#saver.restore(sess,'ckpts_freeze/20000.ckpt')

for tv in tf.trainable_variables():
    if tv.name.find('weights')>0:
        #weight is h, w, bch, tch
        w = np.zeros(tv.get_shape().as_list())
        if tv.name.find('conv1')>-1:
            w[:,:,0,0] = [[1,1,1],
                          [1,0,0],
                          [1,0,0]]

            w[:,:,0,1] = [[1,0,0],
                          [1,0,0],
                          [1,1,1]]

            w[:,:,0,2] = [[-3,-3,-3],
                          [1,1,1],
                          [-3,-3,-3]]

        elif tv.name.find('conv2')>-1:
            w[:,:,0,0] = [[0,1,0],
                          [0,1,0],
                          [0,0,0]]

            w[:,:,1,0] = [[0,0,0],
                          [0,1,0],
                          [0,1,0]]

            """
            w[:,:,2,0] = [[0,0,0],
                          [0,-7,0],
                          [0,0,0]]
            """
            w[:,:,2,0] = [[0,-7,0],
                          [0,-7,0],
                          [0,-7,0]]

            w[:,:,0,1] = [[0,-1,0],
                          [0,-1,0],
                          [0,0,0]]

            w[:,:,1,1] = [[0,0,0],
                          [0,-1,0],
                          [0,-1,0]]

            w[:,:,2,1] = [[0,7,0],
                          [0,7,0],
                          [0,7,0]]

        elif tv.name.find('conv3')>-1:
            w[:,:,0,0] = [[0,1,0],
                          [0,1,0],
                          [0,0,0]]

            w[:,:,1,0] = [[0,0,0],
                          [0,-7,0],
                          [0,-7,0]]

            w[:,:,0,1] = [[0,-1,0],
                          [0,-1,0],
                          [0,0,0]]

            w[:,:,1,1] = [[0,0,0],
                          [0,7,0],
                          [0,7,0]]

        elif tv.name.find('conv4')>-1:
            w[:,:,0,0] = [[0,0,0],
                          [0,4,0],
                          [0,0,0]]

            w[:,:,1,0] = [[0,0,0],
                          [0,-8,0],
                          [0,0,0]]

            w[:,:,0,1] = [[0,0,0],
                          [0,-4,0],
                          [0,0,0]]

            w[:,:,1,1] = [[0,0,0],
                          [0,8,0],
                          [0,0,0]]
        w+=0.01
        w*=0.01
        sess.run(tv.assign(w))
        
    else:
        trainvs.append(tv)

#saver.save(sess,'freeze.ckpt')
#saver.restore(sess,'freeze.ckpt')

for tv in tf.trainable_variables():
    print tv.name, sess.run(tv)


for tv in trainvs:
    print tv.name


for i in xrange(10000):
    x, y= create_batch(batch_size)
    #normalize and centralize the batch

    if i%100 == 0:
        train_accuracy = loss.eval(feed_dict={inputs:x, labels:y})
        loval = logits.eval(feed_dict={inputs:x, labels:y})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        saver.save(sess, 'ckpts_freeze3/%d.ckpt'%(i))
    train_step.run(feed_dict={inputs:x, labels:y})
