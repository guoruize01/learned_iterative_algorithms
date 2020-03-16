#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import math
import tensorflow as tf

class Generator(object):
    def __init__(self,A,**kwargs):
        self.A = A
        M,N = A.shape
        vars(self).update(kwargs)
        self.x_ = tf.placeholder( tf.float32,(N,None),name='x' )
        self.y_ = tf.placeholder( tf.float32,(M,None),name='y' )

class TFGenerator(Generator):
    def __init__(self,**kwargs):
        Generator.__init__(self,**kwargs)
    def __call__(self,sess):
        'generates y,x pair for training'
        return sess.run( ( self.ygen_,self.xgen_ ) )

class NumpyGenerator(Generator):
    def __init__(self,**kwargs):
        Generator.__init__(self,**kwargs)

    def __call__(self,sess):
        'generates y,x pair for training'
        return self.p.genYX(self.nbatches,self.nsubprocs)


def bernoulli_gaussian_trial(M=250,N=500,L=1000,pnz=.1,kappa=None,SNR=40):
    # This function returns an object called prob which contains:
    # the measurement matrix, both numpy array A and TensorFlow constant A_,
    # Tensors xgen, ygen_ which can be used in TensorFlow to generate new training data,
    # numpy arrays xval and yval which are used to evaluate the learned network
    # numpy arrays xinit and yinit, which I am not sure are used at all ???
    # and a scalar noise_var


    A = np.random.normal(size=(M, N), scale=1.0 / math.sqrt(M)).astype(np.float32)

    # A_pnz = 0.1
    # A_sparse = ((np.random.uniform(0, 1, (M, N)) < A_pnz) * A / math.sqrt(A_pnz)).astype(np.float32)
    # A = A_sparse

    # A_0pm1_pnz = 0.1
    # A_0pm1 = ((np.random.uniform(0, 1, (M, N)) < A_0pm1_pnz) * np.sign(A) / math.sqrt(M * A_0pm1_pnz)).astype(np.float32)
    # A = A_0pm1

    # U_helper = np.random.normal(size=(N, N), scale=1.0).astype(np.float32)
    # U, S, Vh = np.linalg.svd(U_helper)
    # # print(U.shape)
    # rows = np.random.permutation(np.arange(N))[0:M]
    # A = U[rows, :] * np.sqrt(N/M)

    col_normalized = False
    if col_normalized:
        A = A / np.sqrt(np.sum(np.square(A), axis=0, keepdims=True))

    if not(kappa is None):
        if kappa >= 1:
            # create a random operator with a specific condition number
            U,_,V = la.svd(A,full_matrices=False)
            s = np.logspace( 0, np.log10( 1/kappa),M)
            A = np.dot( U*(s*np.sqrt(N)/la.norm(s)),V).astype(np.float32)
    A_ = tf.constant(A,name='A')
    prob = TFGenerator(A=A,A_=A_,pnz=pnz,kappa=kappa,SNR=SNR)
    prob.name = 'Bernoulli-Gaussian, random A'

    from scipy.io import loadmat
    W_dict = loadmat('W.mat')
    prob.W = np.transpose(W_dict["W"])

    bernoulli_ = tf.to_float( tf.random_uniform( (N,L) ) < pnz)
    xgen_ = bernoulli_ * tf.random_normal( (N,L) )
    noise_var = pnz*N/M * math.pow(10., -SNR / 10.)
    ygen_ = tf.matmul( A_,xgen_) + tf.random_normal( (M,L),stddev=math.sqrt( noise_var ) )

    prob.xval = ((np.random.uniform( 0,1,(N,L))<pnz) * np.random.normal(0,1,(N,L))).astype(np.float32)
    prob.yval = np.matmul(A,prob.xval) + np.random.normal(0,math.sqrt( noise_var ),(M,L))
    prob.xinit = ((np.random.uniform( 0,1,(N,L))<pnz) * np.random.normal(0,1,(N,L))).astype(np.float32)
    prob.yinit = np.matmul(A,prob.xinit) + np.random.normal(0,math.sqrt( noise_var ),(M,L))
    prob.xgen_ = xgen_
    prob.ygen_ = ygen_
    prob.noise_var = noise_var
    prob.pnz = pnz

    from scipy.io import loadmat
    W_dict = loadmat('W.mat')
    prob.W = np.transpose(W_dict["W"])

    return prob

def one_bit_CS_with_BG_prior(M=500, N=500, L=1000, pnz=.1, kappa=None, SNR=None, tf_floattype = tf.float32):
    # This function returns an object called prob which contains:
    # the measurement matrix, both numpy array A and TensorFlow constant A_,
    # Tensors xgen, ygen_ which can be used in TensorFlow to generate new training data,
    # numpy arrays xval and yval which are used to evaluate the learned network
    # numpy arrays xinit and yinit, which I am not sure are used at all ???
    # and a scalar noise_va

    A = np.random.normal(size=(M, N), scale=1.0 / math.sqrt(M)).astype(np.float32)

    A_ = tf.constant(A, name='A', dtype=tf_floattype)
    prob = TFGenerator(A=A, tf_floattype=tf_floattype, A_=A_,pnz=pnz,kappa=kappa,SNR=SNR)
    prob.name = '1bit CS, BG prior, Gaussian A'
    prob.pnz = pnz
    prob.code_symbol = np.sqrt(prob.pnz*N/M)

    bernoulli_ = tf.to_float( tf.random_uniform( (N,L) ) < pnz)
    xgen_ = bernoulli_ * tf.random_normal( (N,L) )

    xgen_ = tf.dtypes.cast(xgen_, dtype=tf_floattype)

    if SNR is None:
      noise_var = 0
    else:
      # This definition is with correspondence to the MATLAB code
      # Here the SNR is related to P(y)/P(w)
      noise_var = math.pow(10., -SNR / 10.)*prob.code_symbol**2
      # where as in this definition SNR is related to P(x)/P(w)
      # noise_var = pnz * N / M * math.pow(10., -SNR / 10.)

    ygen_ = prob.code_symbol*tf.math.sign(tf.matmul(A_, xgen_)) + tf.random_normal((M, L), stddev=math.sqrt(noise_var), dtype=tf_floattype)
    ygen_ = tf.dtypes.cast(ygen_, dtype=tf_floattype)

    prob.xval = ((np.random.uniform( 0,1,(N,L))<pnz) * np.random.normal(0,1,(N,L))).astype(np.float32)
    prob.y_starval = prob.code_symbol*np.sign(np.matmul(A,prob.xval))
    prob.noise = np.random.normal(0,math.sqrt( noise_var ),(M,L))
    prob.yval = prob.y_starval + prob.noise

    prob.xinit = ((np.random.uniform( 0,1,(N,L))<pnz) * np.random.normal(0,1,(N,L))).astype(np.float32)
    prob.yinit = np.sign(np.matmul(A,prob.xinit)) + np.random.normal(0,math.sqrt( noise_var ),(M,L))

    prob.xgen_ = xgen_
    prob.ygen_ = ygen_
    prob.noise_var = noise_var

    y_star_test = prob.y_starval[:,1]

    y_starval_norm2 = la.norm(prob.y_starval[:,1])**2
    noise_norm2 = la.norm(prob.noise[:,1])**2

    # print('y_star_test size is', y_star_test.size)
    # print('y_starval norm2 is', y_starval_norm2)
    # print('prob.noise var=',prob.noise_var)
    # print('noise norm2 is', noise_norm2)

    SNR = 10*np.log10(y_starval_norm2 / noise_norm2)
    print('SNR=',SNR)

    return prob


def ssc_problem(n=50, L=4, bps=4, MC=1000, SNR_dB=None):
    # This function returns an object called prob which contains:
    # the measurement matrix, both numpy array A and TensorFlow constant A_,
    # Tensors xgen, ygen_ which can be used in TensorFlow to generate new training data,
    # numpy arrays xval and yval which are used to evaluate the learned network
    # numpy arrays xinit and yinit, which I am not sure are used at all ???
    # and a scalar noise_va

    # MC is the number of transmitted messages (Monte Carlo simulations)
    # L is the number of sections
    # bits per section
    # R is the rate of the code, bits per channel use
    # number of channel uses, i.e., number of rows of A


    B = np.power(2, bps)    # B is the size of the section
    N = B * L               # N is the length of a uncoded SSC message, i.e., number of columns of A
    k = L * bps             # number of transmitted bits
    noise_var =1

    A = np.random.normal(size=(n, N), scale=1.0 / math.sqrt(n)).astype(np.float32)

    A_ = tf.constant(A, name='A')
    prob = TFGenerator(A=A, A_=A_, kappa=None, SNR=SNR_dB)
    prob.name = 'SSC, Gaussian A'
    prob.n = n
    prob.L = L
    prob.bps = bps
    prob.SNR_dB = SNR_dB
    prob.B = B
    prob.N = N
    prob.k = k
    prob.P = math.pow(10., SNR_dB / 10.)
    prob.Pl = prob.P/L
    prob.noise_var = noise_var
    prob.sqrtnPl = np.sqrt(n*prob.Pl)



    # Create tf vectors
    messages_ = tf.random.uniform((MC, L), maxval=B, dtype=tf.int32)
    prob.messages_ = messages_

    x_ = tf.one_hot(messages_, depth=B)
    xgen_ = prob.sqrtnPl*tf.transpose(tf.reshape(x_, [MC, N]))
    y_clean_gen_ = tf.matmul(A_, xgen_)
    noise_ = tf.random_normal((n, MC), stddev=math.sqrt(noise_var))
    ygen_ = y_clean_gen_ + noise_

    # Create validation vectors
    messages = np.random.randint(0, B, (L, MC))
    x = np.zeros((N, MC))
    for sample_index in range(MC):
        for i in range(0, L):
            x[i * B + messages[i, sample_index], sample_index] = prob.sqrtnPl

    prob.xval = x
    prob.y_cleanval = np.matmul(A, prob.xval)
    prob.noise = np.random.normal(0, math.sqrt(noise_var), (n, MC))
    prob.yval = np.matmul(A, prob.xval) + prob.noise

    # Uncomment for checking SNR
    y_clean_norm2 = np.mean(la.norm(prob.y_cleanval, axis=0)**2)
    noise_norm2 = np.mean(la.norm(prob.noise, axis=0)**2)
    SNR_empirical = y_clean_norm2/noise_norm2
    SNR_dB_empirical = 10*np.log10(SNR_empirical)
    print('y_clean_norm=',y_clean_norm2)
    print('noise_norm=',noise_norm2)
    print('SNR_empirical=', SNR_empirical)
    print('SNR_dB_empirical=', SNR_dB_empirical)

    # # Not sure if this is needed
    # prob.xinit = ((np.random.uniform( 0,1,(N,L))<pnz) * np.random.normal(0,1,(N,L))).astype(np.float32)
    # prob.yinit = np.sign(np.matmul(A,prob.xinit)) + np.random.normal(0,math.sqrt( noise_var ),(M,L))

    prob.xgen_ = xgen_
    prob.ygen_ = ygen_

    return prob

def block_gaussian_trial(m=128, L=32, B=16, MC=1000, pnz=.1, SNR_dB=20):

    N = B * L  # N is the length of a the unknown block-sparse x
    A = np.random.normal(size=(m, N), scale=1.0 / math.sqrt(m)).astype(np.float32)
    A_ = tf.constant(A, name='A')
    prob = TFGenerator(A=A, A_=A_, kappa=None, SNR=SNR_dB)

    prob.name = 'block sparse, Gaussian A'
    prob.L = L
    prob.B = B
    prob.N = N
    prob.SNR_dB = SNR_dB
    prob.pnz = pnz

    # Create tf vectors
    active_blocks_ = tf.to_float(tf.random_uniform((L, 1, MC)) < pnz)
    ones_ = tf.ones([L, B, MC])

    product_ = tf.multiply(active_blocks_, ones_)
    xgen_ = tf.reshape(product_, [L * B, MC])

    # you should probably change the way noise_var is calculated
    noise_var = pnz * N / m * math.pow(10., -SNR_dB / 10.)
    ygen_ = tf.matmul(A_, xgen_) + tf.random_normal((m, MC), stddev=math.sqrt(noise_var))

    active_blocks_val = (np.random.uniform( 0,1,(L,MC))<pnz).astype(np.float32)
    active_entries_val = np.repeat(active_blocks_val, B, axis=0)
    xval = np.multiply(active_entries_val, np.random.normal(0,1,(N,MC)))
    yval = np.matmul(A,xval) + np.random.normal(0,math.sqrt( noise_var ),(m,MC))

    prob.xgen_ = xgen_
    prob.ygen_ = ygen_
    prob.xval = xval
    prob.yval = yval
    prob.noise_var = noise_var

    return prob


def random_access_problem(which=1):
    from tools import raputil as ru
    if which == 1:
        opts = ru.Problem.scenario1()
    else:
        opts = ru.Problem.scenario2()

    p = ru.Problem(**opts)
    x1 = p.genX(1)
    y1 = p.fwd(x1)
    A = p.S
    M,N = A.shape
    nbatches = int(math.ceil(1000 /x1.shape[1]))
    prob = NumpyGenerator(p=p,nbatches=nbatches,A=A,opts=opts,iid=(which==1))
    if which==2:
        prob.maskX_ = tf.expand_dims( tf.constant( (np.arange(N) % (N//2) < opts['Nu']).astype(np.float32) ) , 1)

    _,prob.noise_var = p.add_noise(y1)

    unused = p.genYX(nbatches) # for legacy reasons -- want to compare against a previous run
    (prob.yval, prob.xval) = p.genYX(nbatches)
    (prob.yinit, prob.xinit) = p.genYX(nbatches)
    import multiprocessing as mp
    prob.nsubprocs = mp.cpu_count()
    return prob
