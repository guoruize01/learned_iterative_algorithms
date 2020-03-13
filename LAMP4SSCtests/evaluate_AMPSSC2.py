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


MC = 1000           # MC is the number of transmitted messages (Monte Carlo simulations)
L = 32                # L is the number of sections
bps = 4               # bits per section
R = 1.0               # R is the rate of the code, bits per channel use
n = int(L*bps/R)      # number of channel uses, i.e., number of rows of A

# Create the basic problem structure.
prob = problems.ssc_problem(n=n, L=L, bps=bps, MC=MC, SNR_dB=8)


# # build a LAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
layers = networks.build_LAMP4SSC(prob,T=6,untied=False)
print('Building layers ... done')


# plan the learning
training_stages = train.setup_LAMP4SSCtraining(layers,prob,trinit=1e-3, refinements=(.5,) )
# training_stages = train.setup_LAMP4SSCtraining(layers,prob,trinit=1e-3,refinements=(.5,.1,.01) )
print('Plan the learning ... done')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

# state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer
#
# must use this same Session to perform all training
# if we start a new Session, things would replay and we'd be training with our validation set (no no)
#
# done = state.get('done', [])
# log = str(state.get('log', ''))

for name, xhat_, loss_, nmse_, ser_, train_, var_list in training_stages:
    nmse, ser = sess.run([nmse_, ser_], feed_dict={prob.y_: prob.yval, prob.x_: prob.xval})
    print(name, '\tnmse=', nmse, '\tnmse/dB=', 10 * np.log10(nmse), '\tser=', ser)

sess.close()