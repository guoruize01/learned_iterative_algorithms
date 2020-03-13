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

L=10000
M=250
N=500
SNR=20
pnz=.1
untied=False
T=8
shrink='bg'

# Create the basic problem structure.
prob = problems.bernoulli_gaussian_trial(kappa=None,M=M,N=N,L=L,pnz=pnz,SNR=SNR) #a Bernoulli-Gaussian x, noisily observed through a random matrix
#prob = problems.random_access_problem(2) # 1 or 2 for compressive random access or massive MIMO
print('Problem created ...')
print('A is:')
print(prob.A)

# from scipy.io import savemat
# # W = np.load(config.W)
# dict = dict(D=prob.A)
# savemat( 'D.mat', dict, oned_as='column' )


# build a LAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
layers = networks.build_LAMP(prob,T=T,shrink=shrink,untied=untied)
print('Building layers ... done')


nmse_arrray = []
mse_arrray = []
sigma2_array = []

# Evaluating (fixed-)GAMP in TensorFlow
# For debugging purposes I initialize a session here - Intialize the Session
sess = tf.Session()
y,x_true = prob(sess)


sess.run(tf.global_variables_initializer())

for name, xhat_, rvar_, var_list in layers:

    nmse_denom_ = tf.nn.l2_loss(prob.x_)
    nmse_ = tf.nn.l2_loss( xhat_ - prob.x_) / nmse_denom_

    mse_ = 2* tf.nn.l2_loss(xhat_ - prob.x_) / (L*N)

    rvar_mean_ = tf.reduce_mean(rvar_)

    x_hat, nmse, mse, rvar_mean = sess.run([xhat_, nmse_, mse_, rvar_mean_], feed_dict={prob.y_: y, prob.x_: x_true})

    if "non-linear T=" in name:
        nmse_arrray.append(nmse)
        mse_arrray.append(mse)
        sigma2_array.append(rvar_mean)

    print(name, '\tnmse=', nmse,'\tNMSE/dB=',10*np.log10(nmse),'\tMSE/dB=',10*np.log10(mse), '\t sigma2/dB=',10*np.log10(rvar_mean))


sess.close()

print('nmse/dB=', 10*np.log10(nmse_arrray))
print('mse/dB=', 10*np.log10(mse_arrray))
print('sigma2/dB=', 10*np.log10(sigma2_array))