#!/usr/bin/python
import cv2
import numpy as np
from skimage import io, draw
import tensorflow.contrib.slim as slim
import tensorflow as tf
from sample_maker import create_batch, create_neg, create_pos
from sc_manual import sc
import os, sys
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
losses = []
train_steps = []
for j in range(1,4):
    loss = tf.losses.mean_squared_error(labels,activations['conv%d'%j],weights=10)
    #loss = tf.losses.softmax_cross_entropy(labels,activations['conv%d'%j],weights=10)
    losses.append(loss)
    train_steps.append(tf.train.AdamOptimizer(1e-6).minimize(loss, var_list=trr))

print losses
print train_steps

sess.run(tf.global_variables_initializer())
saver.restore(sess,'ckpts_manual/1.ckpt')

for j in range(1,4):
    tvs = tf.trainable_variables()
    print sess.run(tvs[0])[:,:,0,1]
    posref = sess.run([activations], feed_dict={inputs:poss})[0]
    negref = sess.run([activations], feed_dict={inputs:negs})[0]
    for i in xrange(10000):
        x, y= create_batch(batch_size, noise=True)
        shape = posref['conv%d'%j].shape[1:]
        lb = np.zeros([batch_size, shape[0], shape[1], shape[2]])
        for k in xrange(batch_size):
            lb[k,:,:,:] = posref['conv%d'%j][0,:,:,:] if y[k,0,0,0] == 1 else negref['conv%d'%j][0,:,:,:]
        #normalize and centralize the batch
        #x = x/255
        if i%100 == 0:
            train_accuracy = losses[j-1].eval(feed_dict={inputs:x, labels:lb})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            if train_accuracy == 0:
                break
            
        train_steps[j-1].run(feed_dict={inputs:x, labels:lb})
    tvs = tf.trainable_variables()
    print sess.run(tvs[0])[:,:,0,1]


    """
    print sess.run(tvs[0])[:,:,0,1]
    assert(False)
    """

    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
    train_tensor = slim.learning.create_train_op(total_loss, optimizer)
    slim.learning.train(train_tensor, train_log_dir)
    """
saver.save(sess, 'ckpts_manual_noise_gauss/1.ckpt')
