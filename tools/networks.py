#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import math

import tensorflow as tf
import tensorflow_probability as tfp
import tools.shrinkage as shrinkage

def build_LISTA(prob,T,initial_lambda=.1,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    assert not untied,'TODO: untied'
    eta = shrinkage.simple_soft_threshold
    layers = []
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_0')
    By_ = tf.matmul( B_ , prob.y_ )
    layers.append( ('Linear',By_,None) )

    initial_lambda = np.array(initial_lambda).astype(np.float32)
    if getattr(prob,'iid',True) == False:
        # create a parameter for each coordinate in x
        initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, lam0_)
    layers.append( ('LISTA T=1',xhat_, (lam0_,) ) )
    for t in range(1,T):
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = eta( tf.matmul(S_,xhat_) + By_, lam_ )
        layers.append( ('LISTA T='+str(t+1),xhat_,(lam_,)) )
    return layers

def build_LBISTA(prob,T,initial_lambda=.1,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN

    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA <- DELETE if unnecessary
    untied          - flag for tied or untied case

    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """

    layers = []

    """
    # check other functions in this file (e.g., build_LISTA and build_LAMP4SSC) to implement LBISTA network
    # send me questions if needed
    """

    return layers


def build_LAMP(prob,T,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink, prob)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    M,N = A.shape

    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')

    By_ = tf.matmul( B_ , prob.y_ )
    layers.append( ('Linear',By_,tf.constant(0.0),tf.constant(0.0),None) )

    if getattr(prob,'iid',True) == False:
        # set up individual parameters for every coordinate
        theta_init = theta_init*np.ones( (N,1),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rhat_ = By_
    rvar_ = tf.reduce_sum(tf.square(prob.y_),0) * OneOverM

    (xhat_,dxdr_) = eta( rhat_, rvar_ , theta_ )
    layers.append( ('LAMP-{0} non-linear T=1'.format(shrink),xhat_,rhat_,rvar_,(theta_,) ) )

    vt_ = prob.y_
    for t in range(1,T):
        if len(dxdr_.get_shape())==2:
            dxdr_ = tf.reduce_mean(dxdr_,axis=0)
        bt_ = dxdr_ * NOverM
        vt_ = prob.y_ - tf.matmul( prob.A_ , xhat_ ) + bt_ * vt_
        rvar_ = tf.reduce_sum(tf.square(vt_),0) * OneOverM
        theta_ = tf.Variable(theta_init,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat_ = xhat_ + tf.matmul(B_,vt_)
            layers.append( ('LAMP-{0} linear T={1}'.format(shrink,t+1),rhat_ ,(B_,) ) )
        else:
            rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_,dxdr_) = eta( rhat_ ,rvar_ , theta_ )
        layers.append( ('LAMP-{0} non-linear T={1}'.format(shrink,t+1),xhat_,rhat_,rvar_,(theta_,) ) )

    return layers

def build_GLAMP(prob,T,shrink,untied):
    """
    Builds a GLAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink, prob)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    M,N = A.shape


    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')

    By_ = tf.matmul( B_ , prob.y_ )
    layers.append( ('Linear',By_,tf.constant(0.0),None) )

    if getattr(prob,'iid',True) == False:
        # set up individual parameters for every coordinate
        theta_init = theta_init*np.ones( (N,1),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reduce_sum(tf.square(prob.y_),0) * OneOverM
    (xhat_,dxdr_) = eta( By_,rvar_ , theta_ )
    layers.append( ('LAMP-{0} non-linear T=1'.format(shrink),xhat_,rvar_,(theta_,) ) )

    vt_ = prob.y_
    for t in range(1,T):
        if len(dxdr_.get_shape())==2:
            dxdr_ = tf.reduce_mean(dxdr_,axis=0)
        bt_ = dxdr_ * NOverM
        vt_ = prob.y_ - tf.matmul( prob.A_ , xhat_ ) + bt_ * vt_
        rvar_ = tf.reduce_sum(tf.square(vt_),0) * OneOverM
        theta_ = tf.Variable(theta_init,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat_ = xhat_ + tf.matmul(B_,vt_)
            layers.append( ('LAMP-{0} linear T={1}'.format(shrink,t+1),rhat_ ,(B_,) ) )
        else:
            rhat_ = xhat_ + tf.matmul(B_,vt_)

        D_1_exp_helper_ = tf.reshape(tf.stack([tf.eye(N), -tf.eye(N)]), [2 * N, -1])
        # D_1_exp_ = tf.constant(D_1_exp_helper_, dtype=tf.float32, name='D_1_')

        # lam = theta_[0] * tf.sqrt(rvar_)

        # b_1_ = tf.Variable(tf.zeros([2*N, 1]), dtype=tf.float32, name='b_1_')
        # b_1_ = tf.Variable(tf.ones([2 * N, 1]), dtype=tf.float32, name='b_1_')

        lam = theta_[0] * tf.sqrt(rvar_)
        scale = theta_[1]

        b_1_helper_ = tf.reshape(tf.stack([tf.ones([N, 1]), tf.ones([N, 1])]), [2 * N, -1])
        # b_1_ = tf.Variable(b_1_helper_, dtype=tf.float32, name='b_1_')


        rhat_expanded_1_  = tf.nn.relu(tf.matmul(D_1_exp_helper_, rhat_) - lam * b_1_helper_)

        dxdr_ = 2*tf.reduce_mean(tf.to_float(rhat_expanded_1_ > 0), 0)

        D_1_com_helper_ = tf.transpose(tf.reshape(tf.stack([tf.eye(N), -tf.eye(N)]), [2 * N, -1]))
        # D_1_com_ = tf.constant(D_1_com_helper_, dtype=tf.float32, name='B_com')

        xhat_ = tf.matmul(D_1_com_helper_, rhat_expanded_1_)

        xhat_ = scale * xhat_
        dxdr_ = scale * dxdr_

        layers.append( ('LAMP-{0} non-linear T={1}'.format(shrink,t+1),xhat_,rvar_,(theta_,) ) )

    return layers

def build_ALAMP(prob,T,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink, prob)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    M,N = A.shape

    # B = A.T
    B = prob.W
    B_ = tf.constant(B, dtype=tf.float32)

    By_ = tf.matmul( B_ , prob.y_ )

    if getattr(prob,'iid',True) == False:
        # set up individual parameters for every coordinate
        theta_init = theta_init*np.ones( (N,1),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reduce_sum(tf.square(prob.y_),0) * OneOverM
    (xhat_,dxdr_) = eta( By_,rvar_ , theta_ )
    layers.append( ('LAMP-{0} non-linear T=1'.format(shrink),xhat_,rvar_,(theta_,) ) )

    vt_ = prob.y_
    for t in range(1,T):
        if len(dxdr_.get_shape())==2:
            dxdr_ = tf.reduce_mean(dxdr_,axis=0)
        bt_ = dxdr_ * NOverM
        vt_ = prob.y_ - tf.matmul( prob.A_ , xhat_ ) + bt_ * vt_
        rvar_ = tf.reduce_sum(tf.square(vt_),0) * OneOverM
        theta_ = tf.Variable(theta_init,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat_ = xhat_ + tf.matmul(B_,vt_)
            layers.append( ('LAMP-{0} linear T={1}'.format(shrink,t+1),rhat_ ,(B_,) ) )
        else:
            rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_,dxdr_) = eta( rhat_ ,rvar_ , theta_ )
        layers.append( ('LAMP-{0} non-linear T={1}'.format(shrink,t+1),xhat_,rvar_,(theta_,) ) )

    return layers

def build_LGAMP(prob, T, shrink, untied, tf_floattype=tf.float32):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    F, F_theta_init = shrinkage.get_shrinkage_function(shrink, prob)
    G, G_theta_init = shrinkage.get_shrinkage_function('bg', prob)

    F_theta_ = tf.constant(F_theta_init, dtype=tf_floattype, name='F_theta_0')
    G_theta_ = tf.Variable(G_theta_init, dtype=tf_floattype, name='G_theta_0')

    layers = []

    A = prob.A
    A_ = tf.dtypes.cast(prob.A_, dtype=tf_floattype)
    M, N = A.shape
    AT = A.T
    # This line causes numerical instability
    # AT = A.T / (1.01 * la.norm(A,2)**2)
    AT_ = tf.Variable(AT, dtype=tf_floattype, name='AT')
    A_A = np.multiply(A, A)
    A_A_ = tf.constant(A_A, dtype=tf_floattype, name='AA')
    A_AT_ = tf.transpose(A_A_)

    sigma_x2 = tf.constant(1, dtype=tf_floattype)
    sigmaw2 = tf.constant(prob.noise_var, dtype=tf_floattype)
    code_scale = prob.code_symbol

    xhat_ = 0 * tf.matmul(AT_, prob.y_)
    xvar_ = tf.constant(prob.pnz, dtype=tf_floattype) * sigma_x2 * (1 + xhat_)
    s_ = 0 * prob.y_

    for t in range(0, T):
        pvar_ = tf.matmul(A_A_, xvar_)
        p_ = tf.matmul(A_, xhat_) - pvar_ * s_

        (s_, svar_) = F(prob.y_, p_, pvar_, sigmaw2, M, F_theta_, code_scale)

        svar_rm = tf.reduce_mean(svar_, axis=0)
        svar_ = tf.matmul(1 + 0 * svar_, tf.matrix_diag(svar_rm))

        rvar_ = tf.math.reciprocal(tf.matmul(A_AT_, svar_))

        rvar_rm = tf.reduce_mean(rvar_, axis=0)
        rvar_ = tf.matmul(1 + 0 * rvar_, tf.matrix_diag(rvar_rm))

        r_ = xhat_ + tf.matmul(AT_, s_) * rvar_

        (xhat_, dxdr_) = G(r_, rvar_, G_theta_)
        xvar_ = rvar_ * dxdr_  # Rangan GAMP, eq 8b

        # layers.append( ('LGAMP T={0}'.format(t+1),xhat_,xvar_,r_,rvar_,s_,svar_,p_,pvar_,(G_theta_,) ) )
        layers.append(('LGAMP T={0}'.format(t + 1), xhat_, (G_theta_,)))

    return layers

def build_LAMP4SSC(prob,T,untied, alg_version):
    L = prob.L

    theta_init = 1.
    print('theta_init=' + repr(theta_init))
    theta_ = tf.Variable(theta_init, dtype=tf.float32, name='theta_0')

    layers = []
    A = prob.A
    A_ = prob.A_
    n, N = A.shape
    B = A.T
    B_ = tf.Variable(B, dtype=tf.float32, name='B_0')
    z_ = prob.y_

    if alg_version == 'tied' or alg_version == 'untied':
        s_ = tf.matmul(B_, z_)
        # var_list: Defaults to the list of variables collected in the graph under the key GraphKeys.TRAINABLE_VARIABLES
        layers.append(('LAMP for SSC Linear-B\t', s_, None))
    elif "tied S" in alg_version:
        # initalization with randS = tf.random_normal((N, N), stddev=1.0 / math.sqrt(N)) is simply bad
        S_ = tf.Variable(tf.eye(N), dtype=tf.float32, name='S')
        s_amp_ = tf.matmul(B_, z_)
        s_ = tf.matmul(S_, s_amp_)
        layers.append(('LAMP for SSC Linear-B,Linear-S\t', s_, None))


    tau_ = theta_ * tf.sqrt(tf.reduce_sum(z_ ** 2, axis=0) / n)
    s1_ = prob.sqrtnPl * tf.matmul((s_ - prob.sqrtnPl), tf.linalg.diag(1 / (tau_ ** 2)))
    s2_ = tf.reshape(tf.transpose(s1_), (-1, prob.B))
    beta_rs_ = prob.sqrtnPl * tf.nn.softmax(s2_, axis=1)
    beta_ = tf.transpose(tf.reshape(beta_rs_, [-1, N]))

    layers.append( ('LAMP for SSC non-linear T= 1\t', beta_, (theta_,)) )

    for t in range(1,T):

        if "no Onsager" in  alg_version:
            print('Guys no Onsager term!')
            z_ = prob.y_ - tf.matmul(A_, beta_)
        else:
            ons_ = tf.matmul(z_, tf.linalg.diag(1 / (tau_ ** 2)))
            ons_ = tf.matmul(ons_, tf.linalg.diag(prob.P - tf.norm(beta_, axis=0) ** 2 / n))
            z_ = prob.y_ - tf.matmul(A_, beta_) + ons_

        if alg_version == 'tied':
            s_ = beta_ + tf.matmul(B_, z_)
        elif alg_version == 'untied':
            B_ = tf.Variable(B, dtype=tf.float32, name='B_' + str(t))
            s_ = beta_ + tf.matmul(B_, z_)
            layers.append(('LAMP for SSC linear-B T={:2d}'.format(t + 1), s_, (B_,)))
        elif "tied LAMP tied S" in alg_version:
            print('tied LAMP tied S is in alg_version')
            s_amp_ = beta_ + tf.matmul(B_, z_)
            s_ = tf.matmul(S_, s_amp_)
        elif "tied LAMP untied S" in alg_version:
            print('untied LAMP tied S is in alg_version')
            S_ = tf.Variable(tf.eye(N), dtype=tf.float32, name='S_' + str(t))
            s_amp_ = beta_ + tf.matmul(B_, z_)
            s_ = tf.matmul(S_, s_amp_)
            layers.append(('LAMP for SSC linear-S T={:2d}'.format(t + 1), s_, (S_,)))
        else:
            s_ = beta_ + tf.matmul(B_, z_)
            print('Something is wrong with alg_version in build_LAMP4SSC')

        # if untied:
        #     B_exp_tail_ = tf.random_normal((N, n), stddev=1.0 / math.sqrt(n))
        #     # B_exp_tail_ = tf.zeros([N, n], tf.float32)
        #     B_exp_helper_ = tf.reshape(tf.stack([B, B_exp_tail_]), [2 * N, -1])
        #     B_exp_ = tf.Variable(B_exp_helper_, dtype=tf.float32, name='B_exp')
        #
        #     # B_com_tail_ = tf.zeros([N, N], tf.float32)
        #     B_com_tail_ = tf.random_normal((N, N), stddev=1.0 / math.sqrt(n))
        #     B_com_helper_ = tf.transpose(tf.reshape(tf.stack([tf.eye(N), B_com_tail_]), [2 * N, -1]))
        #     B_com_ = tf.Variable(B_com_helper_, dtype=tf.float32, name='B_com')
        #
        #     s_ = beta_ + tf.matmul(B_com_, tf.nn.relu(tf.matmul(B_exp_, z_)))
        #
        #     layers.append(('LAMP for SSC non-linear-B-expcom T={0}'.format(t + 1), s_, (B_com_, B_exp_)))

        # if untied:
        #     # B_exp_tail_ = tf.random_normal((N, n), stddev=1.0 / math.sqrt(n))
        #     B_exp_tail_ = tf.zeros([N, n], tf.float32)
        #     B_exp_helper_ = tf.reshape(tf.stack([B, B_exp_tail_]), [2*N,-1])
        #     B_exp_ = tf.Variable(B_exp_helper_, dtype=tf.float32,name='B_exp')
        #
        #     B_inter = np.random.normal(size=(2*N, 2*N), scale=1.0 / math.sqrt(2*N)).astype(np.float32)
        #     B_inter_ = tf.Variable(B_inter, name='B_inter')
        #
        #     # B_com_tail_ = tf.zeros([N, N], tf.float32)
        #     B_com_tail_ = tf.random_normal((N, N), stddev=1.0 / math.sqrt(n))
        #     B_com_helper_ = tf.transpose(tf.reshape(tf.stack([tf.eye(N), B_com_tail_]), [2 * N, -1]))
        #     B_com_ = tf.Variable(B_com_helper_, dtype=tf.float32, name='B_com')
        #
        #     s_ = beta_ + tf.matmul(B_com_, tf.nn.relu(tf.matmul(B_inter_, tf.nn.relu(tf.matmul(B_exp_, z_)))))
        #
        #     layers.append( ('LAMP for SSC non-linear-B-expcom T={:2d}'.format(t+1),s_ ,(B_com_,B_inter_,B_exp_) ) )
        # else:
        #     s_ = beta_ + tf.matmul(B_, z_)


        theta_ = tf.Variable(theta_init, name='theta_' + str(t))
        tau_ = theta_ * tf.sqrt(tf.reduce_sum(z_ ** 2, axis=0) / n)

        s1_ = prob.sqrtnPl * tf.matmul((s_ - prob.sqrtnPl), tf.linalg.diag(1 / (tau_ ** 2)))
        s2_ = tf.reshape(tf.transpose(s1_), (-1, prob.B))
        beta_rs_ = prob.sqrtnPl * tf.nn.softmax(s2_, axis=1)

        beta_ = tf.transpose(tf.reshape(beta_rs_, [-1, N]))

        layers.append(('LAMP for SSC non-linear T={:2d}\t'.format(t + 1), beta_, (theta_,)))

        # layers.append(('LAMP for SSC non-linear T={0}'.format(t + 1), beta_, z_, tau_, Bz_, s1_, s2_, beta_rs_, None))

    return layers

def build_LVAMP(prob,T,shrink):
    """
    Build the LVMAP network with an SVD parameterization.
    Learns the measurement noise variance and nonlinearity parameters
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    M,N = A.shape
    AA = np.matmul(A,A.T)
    s2,U = la.eigh(AA)  # this is faster than svd, but less precise if ill-conditioned
    s = np.sqrt(s2)
    V = np.matmul( A.T,U) / s
    print('svd reconstruction error={nmse:.3f}dB'.format(nmse=20*np.log10(la.norm(A-np.matmul(U*s,V.T))/la.norm(A) ) ) )
    assert np.allclose( A, np.matmul(U*s,V.T),rtol=1e-4,atol=1e-4)
    V_ = tf.constant(V,dtype=tf.float32,name='V')

    # precompute some tensorflow constants
    rS2_ = tf.constant( np.reshape( 1/(s*s),(-1,1) ).astype(np.float32) )  # reshape to (M,1) to allow broadcasting
    #rj_ = tf.zeros( (N,L) ,dtype=tf.float32)
    rj_ = tf.zeros_like( prob.x_)
    taurj_ =  tf.reduce_sum(prob.y_*prob.y_,0)/(N)
    logyvar_ = tf.Variable( 0.0,name='logyvar',dtype=tf.float32)
    yvar_ = tf.exp( logyvar_)
    ytilde_ = tf.matmul( tf.constant( ((U/s).T).astype(np.float32) ) ,prob.y_)  # inv(S)*U*y
    Vt_ = tf.transpose(V_)

    xhat_ = tf.constant(0,dtype=tf.float32)
    for t in range(T):  # layers 0 thru T-1
        # linear step (LMMSE estimation and Onsager correction)
        varRat_ = tf.reshape(yvar_/taurj_,(1,-1) ) # one per column
        scale_each_ = 1/( 1 + rS2_*varRat_ ) # LMMSE scaling individualized per element {singular dimension,column}
        zetai_ = N/tf.reduce_sum(scale_each_,0) # one per column  (zetai_ is 1/(1-alphai) from Phil's derivation )
        adjust_ = ( scale_each_*(ytilde_ - tf.matmul(Vt_,rj_))) * zetai_ #  adjustment in the s space
        ri_ = rj_ + tf.matmul(V_, adjust_ )  # bring the adjustment back into the x space and apply it
        tauri_ = taurj_*(zetai_-1) # adjust the variance

        # non-linear step
        theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_'+str(t))
        xhat_,dxdr_ = eta(ri_,tauri_,theta_)
        if t==0:
            learnvars = None # really means "all"
        else:
            learnvars=(theta_,)
        layers.append( ('LVAMP-{0} T={1}'.format(shrink,t+1),xhat_, learnvars ) )

        if len(dxdr_.get_shape())==2:
            dxdr_ = tf.reduce_mean(dxdr_,axis=0)
        zetaj_ = 1/(1-dxdr_)
        rj_ = (xhat_ - dxdr_*ri_)*zetaj_ # apply Onsager correction
        taurj_ = tauri_*(zetaj_-1) # adjust the variance

    return layers

def build_LVAMP_dense(prob,T,shrink,iid=False):
    """ Builds the non-SVD (i.e. dense) parameterization of LVAMP
    and returns a list of trainable points(name,xhat_,newvars)
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    layers=[]
    A = prob.A
    M,N = A.shape

    Hinit = np.matmul(prob.xinit,la.pinv(prob.yinit) )
    H_ = tf.Variable(Hinit,dtype=tf.float32,name='H0')
    xhat_lin_ = tf.matmul(H_,prob.y_)
    layers.append( ('Linear',xhat_lin_,None) )

    if shrink=='pwgrid':
        theta_init = np.linspace(.01,.99,15).astype(np.float32)
    vs_def = np.array(1,dtype=np.float32)
    if not iid:
        theta_init = np.tile( theta_init ,(N,1,1))
        vs_def = np.tile( vs_def ,(N,1))

    theta_ = tf.Variable(theta_init,name='theta0',dtype=tf.float32)
    vs_ = tf.Variable(vs_def,name='vs0',dtype=tf.float32)
    rhat_nl_ = xhat_lin_
    rvar_nl_ = vs_ * tf.reduce_sum(prob.y_*prob.y_,0)/N

    xhat_nl_,alpha_nl_ = eta(rhat_nl_ , rvar_nl_,theta_ )
    layers.append( ('LVAMP-{0} T={1}'.format(shrink,1),xhat_nl_, None ) )
    for t in range(1,T):
        alpha_nl_ = tf.reduce_mean( alpha_nl_,axis=0) # each col average dxdr

        gain_nl_ = 1.0 /(1.0 - alpha_nl_)
        rhat_lin_ = gain_nl_ * (xhat_nl_ - alpha_nl_ * rhat_nl_)
        rvar_lin_ = rvar_nl_ * alpha_nl_ * gain_nl_

        H_ = tf.Variable(Hinit,dtype=tf.float32,name='H'+str(t))
        G_ = tf.Variable(.9*np.identity(N),dtype=tf.float32,name='G'+str(t))
        xhat_lin_ = tf.matmul(H_,prob.y_) + tf.matmul(G_,rhat_lin_)

        layers.append( ('LVAMP-{0} lin T={1}'.format(shrink,1+t),xhat_lin_, (H_,G_) ) )

        alpha_lin_ = tf.expand_dims(tf.diag_part(G_),1)

        eps = .5/N
        alpha_lin_ = tf.maximum(eps,tf.minimum(1-eps, alpha_lin_ ) )

        vs_ = tf.Variable(vs_def,name='vs'+str(t),dtype=tf.float32)

        gain_lin_ = vs_ * 1.0/(1.0 - alpha_lin_)
        rhat_nl_ = gain_lin_ * (xhat_lin_ - alpha_lin_ * rhat_lin_)
        rvar_nl_ = rvar_lin_ * alpha_lin_ * gain_lin_

        theta_ = tf.Variable(theta_init,name='theta'+str(t),dtype=tf.float32)

        xhat_nl_,alpha_nl_ = eta(rhat_nl_ , rvar_nl_,theta_ )
        alpha_nl_ = tf.maximum(eps,tf.minimum(1-eps, alpha_nl_ ) )
        layers.append( ('LVAMP-{0}  nl T={1}'.format(shrink,1+t),xhat_nl_, (vs_,theta_,) ) )

    return layers
