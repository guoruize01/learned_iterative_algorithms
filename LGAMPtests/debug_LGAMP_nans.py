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

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# For this function you need to use the line below in networks.build_LGAMP!
# layers.append( ('GLAMP T={0}'.format(t+1),xhat_,dxdr_,r_,rvar_,s_,svar_,p_,pvar_,(G_theta_,) ) )
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------


np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train

# Evaluating (fixed-)GAMP in TensorFlow
# For debugging purposes I initialize a session here - Intialize the Session
sess = tf.Session()

# Create the basic problem structure.
# prob = one_bit_CS_with_BG_prior(kappa=None,M=5,N=8,L=2,pnz=1.00,SNR=2, tf_floattype=tf.float32) #a Bernoulli-Gaussian x, noisily observed through a random matrix
prob = problems.one_bit_CS_with_BG_prior(kappa=None,M=2000,N=512,L=1000,pnz=.0625,SNR=2, tf_floattype=tf.float32) #a Bernoulli-Gaussian x, noisily observed through a random matrix
# prob = one_bit_CS_with_BG_prior(kappa=None,M=512,N=250,L=1000,pnz=.1,SNR=2) #a Bernoulli-Gaussian x, noisily observed through a random matrix
#prob = problems.random_access_problem(2) # 1 or 2 for compressive random access or massive MIMO
print('Problem created ...')

# build a LGAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
layers = networks.build_LGAMP(prob,T=7,shrink='onebitF',untied=True,tf_floattype = tf.float32)
print('Building layers ... done')

y, x_true = prob(sess)

sess.run(tf.global_variables_initializer())


for name, xhat_, var_list in layers:

  x_hat = sess.run(xhat_, feed_dict={prob.y_:y})

  NMSE = la.norm(x_true/la.norm(x_true, axis=0) - x_hat/la.norm(x_hat, axis=0), axis=0)**2
  L = len(NMSE)
  NMSE_no_Nan = NMSE[np.logical_not(np.isnan(NMSE))]

  NMSE_dB = 10*math.log10(np.mean(NMSE_no_Nan))
  print(name, 'NMSE=', NMSE_dB, '\tdB with',L - len(NMSE_no_Nan),'instances of NaN (out of',L,')')



# for name, xhat_, xvar_, r_, rvar_, s_, svar_, p_, pvar_, var_list in layers:
#
#   x_hat, xvar, r, rvar, s, svar, p , pvar = sess.run([xhat_, xvar_, r_, rvar_, s_, svar_, p_, pvar_], feed_dict={prob.y_:y, prob.x_:x_true})
#
#   NMSE = la.norm(x_true/la.norm(x_true, axis=0) - x_hat/la.norm(x_hat, axis=0), axis=0)**2
#   L = len(NMSE)
#   NMSE_no_Nan = NMSE[np.logical_not(np.isnan(NMSE))]
#
#   NMSE_dB = 10*math.log10(np.mean(NMSE_no_Nan))
#   print(name, 'NMSE=', NMSE_dB, '\tdB with',L - len(NMSE_no_Nan),'instances of NaN (out of',L,')')
#
#   if L != len(NMSE_no_Nan):
#     print("Let's do some debugging ...")
#     nan_indices = np.isnan(NMSE)
#
#     p_nan = p[:,nan_indices]
#     pvar_nan = pvar[:,nan_indices]
#     s_nan = s[:,nan_indices]
#     svar_nan = svar[:,nan_indices]
#     r_nan = r[:,nan_indices]
#     rvar_nan = rvar[:,nan_indices]
#     xvar_nan = xvar[:,nan_indices]
#     x_hat_nan = x_hat[:,nan_indices]
#     break

# Close the session
sess.close()