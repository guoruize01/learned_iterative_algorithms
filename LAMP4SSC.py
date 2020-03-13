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

class Parameters(object):
    def __init__(self,MC,**not_obligatory):
        self.MC = MC
        vars(self).update(not_obligatory)

parameters = Parameters(MC = 1000) # MC is the number of transmitted messages (Monte Carlo simulations)
parameters.L = 32                   # L is the number of sections
parameters.bps = 4                  # bits per section
parameters.R = 1.0                  # R is the rate of the code, bits per channel use
parameters.n = int(parameters.L*parameters.bps/parameters.R)
                                    # number of channel uses, i.e., number of rows of A
parameters.SNR_dB = 8               # training and evaluation SNR in dB
parameters.T = 8                    # number of layers of the network/iterations of the algorithm
parameters.Onsager = True           # is the Onsager term included in the calculation of the residual
# type of the loss function in the learning. Possible values 'nmse', 'log loss with probs', 'binary crossentropy'
parameters.loss = 'binary crossentropy'
parameters.untied_B = False         # tied or untied B in LAMP
# 'tied', 'untied', 'tied LAMP tied S', 'tied LAMP untied S', 'tied LAMP tied S no Onsager' ....  'tied LAMP untied S loss=nmse',
parameters.alg_version = 'tied'

# Create the basic problem structure.
prob = problems.ssc_problem(n=parameters.n, L=parameters.L, bps=parameters.bps, MC=parameters.MC, SNR_dB=parameters.SNR_dB)

# # build a LAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
layers = networks.build_LAMP4SSC(prob,T=parameters.T,untied=parameters.untied_B, alg_version=parameters.alg_version)
print('Building layers ... done')


# plan the learning
# training_stages = train.setup_LAMP4SSCtraining(layers,prob,trinit=1e-3, refinements=(.5,) )
training_stages = train.setup_LAMP4SSCtraining(layers,prob,trinit=1e-3,refinements=(.5,.1,.01), parameters = parameters)
print('Plan the learning ... done')

# do the learning (takes a whixle)
print('Do the learning (takes a while)')
sess = train.do_LAMP4SSCtraining(training_stages,prob,'LAMP4SSC.npz',10,100,10)
# sess = train.do_LAMP4SSCtraining(training_stages,prob,'LAMP4SSC.npz',10,10000,500)
# sess = train.do_LAMP4SSCtraining(training_stages,prob,'LAMP4SSC.npz')

# train.plot_estimate_to_test_message(sess, training_stages, prob, 'LAMP_bg_giid.npz' )
# train.test_vector_sizes(sess, training_stages, prob, 'LAMP_bg_giid.npz' )
train.evaluate_LAMP4SSC_nmse(sess, training_stages, prob, 'LAMP4SSC.npz' )

train_vars = train.get_train_variables(sess)

sess.close()