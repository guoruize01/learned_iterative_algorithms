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
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf

np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train


# Create the basic problem structure.
prob = problems.bernoulli_gaussian_trial(kappa=None,M=250,N=500,L=1000,pnz=.1,SNR=40) #a Bernoulli-Gaussian x, noisily observed through a random matrix
#prob = problems.random_access_problem(2) # 1 or 2 for compressive random access or massive MIMO
# print('Problem created ...')
# print('A is:')
# print(prob.A)

filename = 'LAMP_bg_giid.npz'

other = {}
B_t = []
U_t = []
S_t = []
V_t = []
theta_t = []

try:
    filecontent = np.load(filename).items()
    for k, d in filecontent:
        if k.startswith('B_'):
            B_t.append(d)
            print('restoring ' + k + ' is:' + str(d))
        elif k.startswith('theta_'):
            theta_t.append(d)
except IOError:
    pass



A = prob.A
M,N = A.shape
U_A, S_A, Vh_A = np.linalg.svd(A.transpose())

for i in range(len(B_t)):
    print(str(i))
    U, S, Vh = np.linalg.svd( B_t[i])
    U_t.append(U)
    S_t.append(S)
    V_t.append(Vh)


dot_prod = U_A.transpose().dot(U_A)

test = np.amax(dot_prod, axis=0)

stop = 1