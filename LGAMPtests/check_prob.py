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
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
import tensorflow_probability as tfp

np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

from tools import problems

prob = problems.one_bit_CS_with_BG_prior(kappa=None,M=2000,N=512,L=1000,pnz=0.0625,SNR=2, tf_floattype = tf.float32) #a Bernoulli-Gaussian x, noisily observed through a random matrix

print('prob.noise_var=', prob.noise_var, '￿, and should be equal to 0.010095317511683￿')
print('la.norm(prob.yval[:,0])=', la.norm(prob.yval[:,0]), '￿, and should be close to 7.2552')