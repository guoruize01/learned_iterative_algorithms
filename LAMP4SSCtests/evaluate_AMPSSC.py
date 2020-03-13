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


MC = 1000               # MC is the number of transmitted messages (Monte Carlo simulations)
L = 32                  # L is the number of sections
bps = 4                 # bits per section
R = 1.0                 # R is the rate of the code, bits per channel use
n = int(L*bps/R)        # number of channel uses, i.e., number of rows of A
T_max = 20              # max number of iterations/layers
SNR_dB = 8              # training and evaluation SNR in dB

# Create the basic problem structure.
prob = problems.ssc_problem(n=n, L=L, bps=bps, MC=MC, SNR_dB=SNR_dB)


# # build a LAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
layers = networks.build_LAMP4SSC(prob,T=T_max,untied=False)
print('Building layers ... done')

ser_arrray = []
nmse_arrray = []


y,x_true = prob(sess)
x_true_rs = np.reshape(np.transpose(x_true), (-1, prob.B))
messages_true = np.reshape(np.argmax(x_true_rs, 1), (-1, prob.L))

sess.run(tf.global_variables_initializer())

for name, xhat_, var_list in layers:

    xhat_rs_ = tf.reshape(tf.transpose(xhat_), (-1, prob.B))
    messages_hat_ = tf.reshape(tf.arg_max(xhat_rs_, 1), (-1, prob.L))

    nmse_denom_ = tf.nn.l2_loss(prob.x_)
    nmse_ = tf.nn.l2_loss( xhat_ - prob.x_) / nmse_denom_


    messages_hat, x_hat, nmse = sess.run([messages_hat_, xhat_, nmse_], feed_dict={prob.y_: y, prob.x_: x_true})

    error_matrix = 1*np.equal(messages_true, messages_hat, dtype=np.int32)
    SER = 1 - np.mean(error_matrix)
    NMSE_dB = 10 * np.log10(nmse)

    if " non-linear T=" in name:
        nmse_arrray.append(NMSE_dB)
        ser_arrray.append(SER)

    MSE = np.mean((la.norm(x_true  - x_hat , axis=0) ** 2)/(n*prob.P))
    print(name, '\tMSE=', '{:.5f}'.format(MSE),'\tMSE/dB=', '{:7.3f}'.format(NMSE_dB) , '\tSER=', '{:.5f}'.format(SER))


sess.close()

print('SER = [',' ,'.join(['{:.5f}'.format(ser) for ser in ser_arrray]),'];')
print('NMSE/dB = [',' ,'.join(['{:.5f}'.format(nmse) for nmse in nmse_arrray]),'];')