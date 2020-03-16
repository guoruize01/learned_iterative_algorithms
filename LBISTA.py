#!/usr/bin/python
from __future__ import division
from __future__ import print_function
"""
This file serves as an example of how to 
a) select a problem to be solved 
b) select a network type
c) train the network to minimize recovery MSE

It provides a basic skeleton for training a Learned Block-ISTA (LBISTA) network
"""

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!
import tensorflow as tf

np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train

class Parameters(object):
    def __init__(self,MC,**not_obligatory):
        self.MC = MC
        vars(self).update(not_obligatory)

parameters = Parameters(MC = 1000)  # MC is the training batch size
parameters.L = 32                   # L is the number of blocks
parameters.B = 16                   # size of each block
parameters.R = 1.0                  # R is the rate of the code, bits per channel use
parameters.m = 128                  # number of measurements, i.e., number of rows of A
parameters.SNR_dB = 8               # training and evaluation SNR in dB

# Create the basic problem structure.
prob = problems.block_gaussian_trial(m=128, L=32, B=16, MC=1000, pnz=.1, SNR_dB=20) # a Block-Gaussian x, noisily observed through a random matrix


# build a LBISTA network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
layers = networks.build_LBISTA(prob,T=6,initial_lambda=.1,untied=False)

# plan the learning
# if you want to minimize nmse you do not need to reimplement the following two lines. You can just play with the parameters of learning
training_stages = train.setup_training(layers,prob,trinit=1e-3,refinements=(.5,.1,.01) )

# do the learning (takes a while)
sess = train.do_training(training_stages,prob,'LBISTA_block_Gauss_giid.npz')
