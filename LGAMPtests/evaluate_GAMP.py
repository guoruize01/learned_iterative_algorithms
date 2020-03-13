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

# Create the basic problem structure.
prob = problems.one_bit_CS_with_BG_prior(kappa=None,M=2000,N=512,L=1000,pnz=.0625,SNR=2, tf_floattype=tf.float32) #a Bernoulli-Gaussian x, noisily observed through a random matrix
# prob = one_bit_CS_with_BG_prior(kappa=None,M=512,N=250,L=1000,pnz=.1,SNR=2) #a Bernoulli-Gaussian x, noisily observed through a random matrix
#prob = problems.random_access_problem(2) # 1 or 2 for compressive random access or massive MIMO
print('Problem created ...')

# build a LGAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
layers = networks.build_LGAMP(prob,T=7,shrink='onebitF',untied=True,tf_floattype = tf.float32)
print('Building layers ... done')

training_stages = train.setup_LGAMPtraining(layers,prob,trinit=1e-3,refinements=(.5,.1,.01) )
print('Plan the learning ... done')

# # do the learning (takes a while)
# print('Do the learning (takes a while)')
# sess = train.do_training(training_stages,prob,'LGAMP_bg_giid.npz',10,30,5)

sess.run(tf.global_variables_initializer())
train.evaluate_LGAMP_nmse(sess, training_stages, prob, pnz=.0625, SNR=2)

# Close the session
sess.close()
if (sess._closed == True):
    print('The session is now closed')
else:
    print('The session is NOT closed')