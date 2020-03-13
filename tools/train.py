#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import sys
import tensorflow as tf
import time

def save_trainable_vars(sess,filename,**kwargs):
    """save a .npz archive in `filename`  with
    the current value of each variable in tf.trainable_variables()
    plus any keyword numpy arrays.
    """
    save={}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    np.savez(filename,**save)

def load_trainable_vars(sess,filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other={}
    try:
        tv=dict([ (str(v.name),v) for v in tf.trainable_variables() ])
        for k,d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k + ' is:' + str(d))
                sess.run(tf.assign( tv[k], d) )
            else:
                other[k] = d
    except IOError:
        pass
    return other

def get_train_variables(sess):
    """save a .npz archive in `filename`  with
        the current value of each variable in tf.trainable_variables()
        plus any keyword numpy arrays.
        """
    save={}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    
    return save

def setup_training(layer_info,prob, trinit=1e-3,refinements=(.5,.1,.01),final_refine=None ):
    """ Given a list of layer info (name,xhat_,newvars),
    create an output list of training operations (name,xhat_,loss_,nmse_,trainop_ ).
    Each layer_info element will be split into one or more output training operations
    based on the presence of newvars and len(refinements)
    """
    losses_=[]
    nmse_=[]
    trainers_=[]
    assert np.array(refinements).min()>0,'all refinements must be in (0,1]'
    assert np.array(refinements).max()<=1,'all refinements must be in (0,1]'

    maskX_ = getattr(prob,'maskX_',1)
    if maskX_ != 1:
        print('masking out inconsequential parts of signal x for nmse reporting')

    nmse_denom_ = tf.nn.l2_loss(prob.x_ *maskX_)

    tr_ = tf.Variable(trinit,name='tr',trainable=False)
    training_stages=[]
    for name,xhat_,rhat_,rvar_,var_list in layer_info:
        loss_  = tf.nn.l2_loss( xhat_ - prob.x_)
        nmse_  = tf.nn.l2_loss( (xhat_ - prob.x_)*maskX_) / nmse_denom_
        sigma2_ = tf.reduce_mean(rvar_)
        sigma2_empirical_ = tf.reduce_mean((rhat_ - prob.x_)**2)

        se_ = 2 * tf.nn.l2_loss(xhat_ - prob.x_)# to get MSE, divide by / (L * N)

        if var_list is not None:
            train_ = tf.train.AdamOptimizer(tr_).minimize(loss_, var_list=var_list)
            training_stages.append( (name,xhat_,sigma2_,loss_,nmse_,sigma2_empirical_,se_,train_,var_list) )
        for fm in refinements:
            train2_ = tf.train.AdamOptimizer(tr_*fm).minimize(loss_)
            training_stages.append( (name+' trainrate=' + str(fm) ,xhat_,sigma2_,loss_,nmse_,sigma2_empirical_,se_,train2_,()) )
    if final_refine:
        train2_ = tf.train.AdamOptimizer(tr_*final_refine).minimize(loss_)
        training_stages.append( (name+' final refine ' + str(final_refine) ,xhat_,sigma2_,loss_,nmse_,sigma2_empirical_,se_,train2_,()) )

    return training_stages

def binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.
    Arguments:
      target: A tensor with the same shape as `output`.
      output: A tensor.
      from_logits: Whether `output` is expected to be a logits tensor.
          By default, we consider that `output`
          encodes a probability distribution.
    Returns:
      A tensor.
    """
    # Note: nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        epsilon_ = tf.convert_to_tensor(1e-07, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon_, 1 - epsilon_)
        output = tf.log(output / (1 - output))
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

def setup_LAMP4SSCtraining(layer_info,prob, trinit=1e-3,refinements=(.5,.1,.01),final_refine=None, parameters=None ):
    """ Given a list of layer info (name,xhat_,newvars),
    create an output list of training operations (name,xhat_,loss_,nmse_,trainop_ ).
    Each layer_info element will be split into one or more output training operations
    based on the presence of newvars and len(refinements)
    """
    losses_=[]
    nmse_=[]
    trainers_=[]
    assert np.array(refinements).min()>0,'all refinements must be in (0,1]'
    assert np.array(refinements).max()<=1,'all refinements must be in (0,1]'

    nmse_denom_ = tf.nn.l2_loss(prob.x_)

    tr_ = tf.Variable(trinit,name='tr',trainable=False)
    training_stages=[]
    for name,xhat_,var_list in layer_info:
        nmse_  = tf.nn.l2_loss( xhat_ - prob.x_) / nmse_denom_


        xhat_rs_ = tf.reshape(tf.transpose(xhat_), (-1, prob.B))
        messages_hat_ = tf.reshape(tf.arg_max(xhat_rs_, 1), (-1, prob.L))

        xtrue_rs_ = tf.reshape(tf.transpose(prob.x_), (-1, prob.B))
        messages_true_ = tf.reshape(tf.arg_max(xtrue_rs_, 1), (-1, prob.L))

        error_matrix_ = tf.dtypes.cast(tf.math.equal(messages_true_, tf.dtypes.cast(messages_hat_, tf.int64)),
                                       dtype=tf.float32)
        ser_ = 1 - tf.reduce_mean(error_matrix_)

        if parameters.loss == 'nmse':
            loss_ = tf.nn.l2_loss(xhat_ - prob.x_)
        elif parameters.loss == 'log loss with probs':
            xhat_rs_01_ = xhat_rs_ / prob.sqrtnPl
            xtrue_rs_01_ = xtrue_rs_ / prob.sqrtnPl
            # sigmoid_cross_entropy_with_logits loss is not the way to go since 1.) it does not consider a binary problem
            # and 2.) each class is independent and not mutually exclusive. However it is interesting the it still kinda
            # wors just as well as l2_loss. It should be noted that I don't use logits, and use probabilities instead
            loss_ = tf.nn.sigmoid_cross_entropy_with_logits(labels=xtrue_rs_01_, logits=xhat_rs_01_)
        elif parameters.loss == 'binary crossentropy':
            # https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/binary-crossentropy
            loss_ = binary_crossentropy(xtrue_rs_01_, xhat_rs_01_)

        # loss_ =  tf.nn.l2_loss( xhat_rs_01_*(1-xtrue_rs_01_) ) # this is shit

        # name = name + ' ' + parameters.loss


        if var_list is not None:
            train_ = tf.train.AdamOptimizer(tr_).minimize(loss_, var_list=var_list)
            training_stages.append( (name,xhat_,loss_,nmse_,ser_,train_,var_list) )
        for fm in refinements:
            train2_ = tf.train.AdamOptimizer(tr_*fm).minimize(loss_)
            training_stages.append( (name+' trainrate=' + str(fm) ,xhat_,loss_,nmse_,ser_,train2_,()) )
    if final_refine:
        train2_ = tf.train.AdamOptimizer(tr_*final_refine).minimize(loss_)
        training_stages.append( (name+' final refine ' + str(final_refine) ,xhat_,loss_,nmse_,ser_,train2_,()) )

    return training_stages

def setup_LGAMPtraining(layer_info,prob, trinit=1e-3,refinements=(.5,.1,.01),final_refine=None ):
    """ Given a list of layer info (name,xhat_,newvars),
    create an output list of training operations (name,xhat_,loss_,nmse_,trainop_ ).
    Each layer_info element will be split into one or more output training operations
    based on the presence of newvars and len(refinements)
    """
    losses_=[]
    nmse_=[]
    trainers_=[]
    assert np.array(refinements).min()>0,'all refinements must be in (0,1]'
    assert np.array(refinements).max()<=1,'all refinements must be in (0,1]'

    tr_ = tf.Variable(trinit,name='tr',trainable=False)
    training_stages=[]
    for name,xhat_,var_list in layer_info:
        # loss_  = tf.nn.l2_loss( xhat_ - prob.x_)
        loss_  = tf.nn.l2_loss( xhat_/tf.norm(xhat_, axis=0) - prob.x_/tf.norm(prob.x_, axis=0))

        nmse_  = tf.reduce_mean(tf.norm(xhat_/tf.norm(xhat_, axis=0) - prob.x_/tf.norm(prob.x_, axis=0), axis=0)**2)
        # loss_ = nmse_

        if var_list is not None:
            train_ = tf.train.AdamOptimizer(tr_).minimize(loss_, var_list=var_list)
            training_stages.append( (name,xhat_,loss_,nmse_,train_,var_list) )
        for fm in refinements:
            train2_ = tf.train.AdamOptimizer(tr_*fm).minimize(loss_)
            training_stages.append( (name+' trainrate=' + str(fm) ,xhat_,loss_,nmse_,train2_,()) )
    if final_refine:
        train2_ = tf.train.AdamOptimizer(tr_*final_refine).minimize(loss_)
        training_stages.append( (name+' final refine ' + str(final_refine) ,xhat_,loss_,nmse_,train2_,()) )

    return training_stages

def do_training(training_stages,prob,savefile,ivl=10,maxit=1000000,better_wait=5000):
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval) ) )

    state = load_trainable_vars(sess,savefile) # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done=state.get('done',[])
    log=str(state.get('log',''))

    for name,xhat_,rvar_,loss_,nmse_,sigma2_empirical_,se_,train_,var_list in training_stages:
        start = time.time()
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])

        print(name + ' ' + describe_var_list)
        nmse_history=[]
        for i in range(maxit+1):
            if i%ivl == 0:
                nmse = sess.run(nmse_,feed_dict={prob.y_:prob.yval,prob.x_:prob.xval})
                if np.isnan(nmse):
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history,nmse)
                nmse_dB = 10*np.log10(nmse)
                nmsebest_dB = 10*np.log10(nmse_history.min())
                sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i,nmse=nmse_dB,best=nmsebest_dB))
                sys.stdout.flush()
                if i%(100*ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin()-1 # how long ago was the best nmse?
                    if age_of_best*ivl > better_wait:
                        break # if it has not improved on the best answer for quite some time, then move along
            y,x = prob(sess)
            sess.run(train_,feed_dict={prob.y_:y,prob.x_:x} )
        done = np.append(done,name)
        
        end = time.time()
        time_log = 'Took me {totaltime:.3f} minutes, or {time_per_interation:.1f} ms per iteration'.format(totaltime = (end-start)/60, time_per_interation = (end-start)*1000/i)
        print(time_log)
        log =  log+'\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name,nmse=nmse_dB,i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess,savefile,**state)
    return sess

def do_LAMP4SSCtraining(training_stages, prob, savefile, ivl=10, maxit=1000000, better_wait=5000):
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done = state.get('done', [])
    log = str(state.get('log', ''))

    for name, xhat_, loss_, nmse_, ser_, train_, var_list in training_stages:
        start = time.time()
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables()])

        print(name + ' ' + describe_var_list)
        nmse_history = []
        for i in range(maxit + 1):
            if i % ivl == 0:
                nmse, ser = sess.run([nmse_, ser_], feed_dict={prob.y_: prob.yval, prob.x_: prob.xval})
                if np.isnan(nmse):
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history, nmse)
                nmse_dB = 10 * np.log10(nmse)
                nmsebest_dB = 10 * np.log10(nmse_history.min())
                sys.stdout.write(
                    '\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f}) with ser={ser:.6f}'.format(i=i, nmse=nmse_dB, best=nmsebest_dB, ser=ser))
                sys.stdout.flush()
                if i % (100 * ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin() - 1  # how long ago was the best nmse?
                    if age_of_best * ivl > better_wait:
                        break  # if it has not improved on the best answer for quite some time, then move along
            y, x = prob(sess)
            sess.run(train_, feed_dict={prob.y_: y, prob.x_: x})
        done = np.append(done, name)

        end = time.time()
        time_log = '\nTook me {totaltime:.3f} minutes, or {time_per_interation:.1f} ms per iteration\n'.format(
            totaltime=(end - start) / 60, time_per_interation=(end - start) * 1000 / i)
        print(time_log)
        log = log + '\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name, nmse=nmse_dB, i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess, savefile, **state)
    return sess

def evaluate_nmse(sess, training_stages, prob, savefile, pnz=.1, SNR=40, L=1000):
    import math

    A = prob.A
    M,N = A.shape

    noise_var = pnz*N/M * math.pow(10., -SNR / 10.)

    data_set_size = 100;

    xtest = ((np.random.uniform( 0,1,(N,data_set_size))<pnz) * np.random.normal(0,1,(N,data_set_size))).astype(np.float32)
    ytest = np.matmul(A, xtest) + np.random.normal(0,math.sqrt( noise_var ),(M,data_set_size))

    nmse_dB_arrray = []
    mse_dB_arrray = []
    sigma2_dB_array = []
    sigma2_empirical_array = []

    for name, xhat_, sigma2_, loss_, nmse_, sigma2_empirical_, se_, train_, var_list in training_stages:

        if " trainrate=" not in name:
            nmse, se, sigma2, sigma2_empirical = sess.run([nmse_, se_, sigma2_, sigma2_empirical_], feed_dict={prob.y_: ytest, prob.x_: xtest})

            nmse_dB = 10 * np.log10(nmse)
            mse_dB = 10 * np.log10(se/(data_set_size*N))
            sigma2_dB =  10 * np.log10(sigma2)
            sigma2_empirical_dB =  10 * np.log10(sigma2_empirical)
            print('{name} nmse={nmse:.6f} dB'.format(name=name,nmse=nmse_dB))

            nmse_dB_arrray.append(nmse_dB)
            mse_dB_arrray.append(mse_dB)
            sigma2_dB_array.append(sigma2_dB)
            sigma2_empirical_array.append(sigma2_empirical_dB)

    print('nmse/dB=', nmse_dB_arrray)
    print('mse/dB=', mse_dB_arrray)
    print('sigma2/dB=', sigma2_dB_array)
    print('sigma2_empirical/dB=', sigma2_empirical_array)

def evaluate_LAMP4SSC_nmse(sess, training_stages, prob, savefile, SNR=40):
    import math

    L = prob.L
    B = prob.B
    A = prob.A
    n,N = A.shape

    noise_var = 1

    data_set_size = 1000;

    # Create validation vectors
    messages = np.random.randint(0, B, (L, data_set_size))
    xtest = np.zeros((N, data_set_size))
    for sample_index in range(data_set_size):
        for i in range(0, L):
            xtest[i * B + messages[i, sample_index], sample_index] = prob.sqrtnPl

    y_clean = np.matmul(A, xtest)
    noise = np.random.normal(0, math.sqrt(noise_var), (n, data_set_size))
    ytest= y_clean + noise

    ser_arrray = []
    nmse_arrray = []

    for name, xhat_, loss_, nmse_, ser_, train_, var_list in training_stages:

        if " trainrate=" not in name:
            nmse, ser = sess.run([nmse_, ser_], feed_dict={prob.y_: ytest, prob.x_: xtest})
            nmse_dB = 10 * np.log10(nmse)
            print('{name} nmse={nmse:.6f} dB with ser={ser:.6f}'.format(name=name,nmse=nmse_dB,ser=ser))
            nmse_arrray.append(nmse_dB)
            ser_arrray.append(ser)

    print('SER = [', ', '.join(['{:.5f}'.format(ser) for ser in ser_arrray]), '];')
    print('NMSE/dB = [', ', '.join(['{:.5f}'.format(nmse) for nmse in nmse_arrray]), '];')


def evaluate_LGAMP_nmse(sess, training_stages, prob, pnz=.1, SNR=2):
    import math

    A = prob.A
    M, N = A.shape

    scale = prob.code_symbol

    noise_var = math.pow(10., -SNR / 10.) * scale ** 2

    data_set_size = 1000;

    xtest = ((np.random.uniform(0, 1, (N, data_set_size)) < pnz) * np.random.normal(0, 1, (N, data_set_size))).astype(
        np.float32)
    y_starval = scale * np.sign(np.matmul(A, xtest))
    noise = np.random.normal(0, math.sqrt(noise_var), (M, data_set_size))
    ytest = y_starval + noise

    x_true = xtest

    for name, xhat_, rvar_, loss_, nmse_, se_, train_, var_list in training_stages:
        if " trainrate=" not in name:
            # The following 6 lines of code are used in debugging.
            # They provide NMSE and the count of nan occurrences


            # x_hat = sess.run(xhat_, feed_dict={prob.y_:ytest, prob.x_:xtest})
            #
            # NMSE = la.norm(x_true/la.norm(x_true, axis=0) - x_hat/la.norm(x_hat, axis=0), axis=0)**2
            # L = len(NMSE)
            # NMSE_no_Nan = NMSE[np.logical_not(np.isnan(NMSE))]
            # NMSE_dB = 10*math.log10(np.mean(NMSE_no_Nan))
            # print(name, 'NMSE=', NMSE_dB, '\tdB with',L - len(NMSE_no_Nan),'instances of NaN (out of',L,')')

            nmse = sess.run(nmse_, feed_dict={prob.y_: ytest, prob.x_: xtest})
            nmse_dB = 10 * np.log10(nmse)
            print('{name} NMSE= {nmse:.6f} \tdB'.format(name=name, nmse=nmse_dB))