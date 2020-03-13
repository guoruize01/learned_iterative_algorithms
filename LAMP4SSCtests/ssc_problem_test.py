#!/usr/bin/python
from __future__ import division
from __future__ import print_function
"""
This file serves as an example of how to 
a) select a problem to be solved 
b) select a network type
c) train the network to minimize recovery MSE

"""
import numpy as np
import math
import numpy.linalg as la
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf


np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train

# Evaluating (fixed-)GAMP in TensorFlow
# For debugging purposes I initialize a session here - Intialize the Session
sess = tf.Session()

# MC is the number of transmitted messages (Monte Carlo simulations)
MC = 10
# L is the number of sections
L = 4
bps = 4
# B is the size of the section
B = np.power(2, bps)
# R is the rate of the code, bits per channel use
R = 1.0
n = int(L*bps / R)
# N is the length of a uncoded SSC message
N = B * L
noise_var = 1.

# Create the basic problem structure.
prob = problems.ssc_problem(n = n, L=L, bps=bps, MC=MC, SNR_dB=8)


r_ = prob.xgen_ + tf.random_normal((N, MC), stddev=math.sqrt(noise_var))

r_rs_ = tf.reshape(tf.transpose(r_ ), (-1,B))
messages_hat_ = tf.reshape(tf.arg_max(r_rs_, 1), (MC,L))
x_hat_ = tf.one_hot(messages_hat_, depth=B)
x_hat_ = tf.transpose(tf.reshape(x_hat_, [MC, N]))

# BLER_ = 1 - tf.reduce_mean(tf.dtypes.cast(prob.messages_ == messages_hat_, dtype = tf.uint8))
error_matrix_ = tf.dtypes.cast(tf.math.equal(prob.messages_, tf.dtypes.cast(messages_hat_, tf.int32)), dtype = tf.float64)
BLER_ = 1 - tf.reduce_mean(error_matrix_)

sess = tf.Session()

messages, xgen, x_hat, messages_hat, BLER_tf, r, r_rs = sess.run([prob.messages_, prob.xgen_, x_hat_, messages_hat_, BLER_,r_, r_rs_])

BLER = 1 - np.mean((messages==messages_hat))

# print('xgen=\n',xgen)
# print('x_hat=\n',x_hat)
print('xgen.shape=\n',xgen.shape)
print('x_hat.shape=\n',x_hat.shape)
print('r.shape=\n',r.shape)
print('r_rs.shape=\n',r_rs.shape)

print('messages=\n',messages)
print('messages_hat=\n',messages_hat)
print('BLER=\n',BLER)
print('BLER_tf=\n',BLER_tf)


messages, xgen, ygen = sess.run([prob.messages_, prob.xgen_, prob.ygen_])

# print('messages=\n',messages)
# print('xgen=\n',xgen)
# print('ygen=\n',ygen)

# print('xgen.shape=\n',xgen.shape)
# print('ygen.shape=\n',ygen.shape)

sess.close()