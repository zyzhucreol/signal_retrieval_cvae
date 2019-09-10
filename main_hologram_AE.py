#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pulse retrieval example

@author: Zheyuan Zhu
"""
import numpy as np
from scipy.io import savemat
import tensorflow as tf
from Hologram.layers import encoderY, decoder
from tensorflow.keras.layers import Conv2D
from Hologram.forward_model import A_fun
from Hologram.read_data import read_data

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0';

#%% model parameters
args=type('',(),{})()
args_fwdmd=type('',(),{})()
args.batch_size = 20 # training batch size
args.train_iters=20000 # number of training iterations
args.learning_rate=1e-4 # learning rate for optimizer

# encoder/decoder arguments
args.T=2 # number of recurrences
args.enc_size = 8 # number of hidden units (channels) in LSTM_encode output
args.dec_size = 8 # number of hidden units (channels) in LSTM_decode output
args.z_size = 8 # Sampler output size
args.convlstm_kernel_size = 64 # 2D Convolutional LSTM kernel size
args.N_det=64
args.N_obj=64

# file I/O parameters
load_model=True
save_name='AE_hologram_1022'
save_model_path='./models/'+save_name # path to save trained model
load_model_path='./models/AE_hologram_1022' # path to load pre-trained weights, if load_model=True
save_mat_folder='./RcVAE_results/'+save_name+'_' # path to save reconstruction examples
log_path='/media/HDD1/zyz/log_RcVAE_keras/'+save_name # Tensorboard path to log training process

#%% build RcVAE graph
x = tf.placeholder(tf.float32,shape=(None,args.N_obj,args.N_obj,1),name='X')
y = tf.placeholder(tf.float32,shape=(None,args.N_det,args.N_det,1),name='Y')
batch_size_flexible=tf.shape(y)[0]

x_p,mu_p,logsigma_p,sigma_p=[0]*args.T,[0]*args.T,[0]*args.T,[0]*args.T

cond_prior_encode=encoderY(args)

cond_prior_lstm=Conv2D(args.enc_size,(args.convlstm_kernel_size,args.convlstm_kernel_size),padding='same',name='cond_prior_lstm')

dec_lstm=Conv2D(args.enc_size,(args.convlstm_kernel_size,args.convlstm_kernel_size),padding='same',name='dec_lstm')

decode=decoder(args)

ry = cond_prior_encode(y)
h_enc_p = cond_prior_lstm(ry)
h_dec_p = dec_lstm(h_enc_p)
Xp = decode(h_dec_p)
    
#%% Define loss function and optimizer
Y_hat=A_fun(Xp)
Lx=tf.losses.mean_squared_error(labels=x,predictions=Xp,scope='MSEX')
Ly=tf.losses.mean_squared_error(labels=y,predictions=Y_hat,scope='MSEY')
L_hybrid=Lx

# Tensorboard monitoring
tf.summary.scalar('MSE_X',Lx)
tf.summary.scalar('MSE_Y',Ly)
tf.summary.image('Xp',Xp)
tf.summary.image('X_true',x)
tf.summary.image('Y_true',y)
tf.summary.image('Yp',Y_hat)
merged=tf.summary.merge_all()

# optimizer
with tf.name_scope('train'):
    optimizer1=tf.train.AdamOptimizer(args.learning_rate)
    train_op1=optimizer1.minimize(L_hybrid)

# Setup training iterations
fetches=[Lx,Ly,train_op1,merged]
fetches_test=[Lx,Ly,merged]

sess=tf.InteractiveSession()
saver = tf.train.Saver()

X_train,I_train,X_test,I_test=read_data()

# choose to initialize the weights or restore the weights from trained model
if load_model is True:
    saver.restore(sess,load_model_path)
else:
    tf.global_variables_initializer().run()

    model1_writer = tf.summary.FileWriter(log_path,sess.graph)
    model2_writer = tf.summary.FileWriter(log_path+'_test',sess.graph)
    
    #%% Run training iterations
    
    for i in range(0,args.train_iters):
        ind_train = np.random.randint(0,np.size(I_train,axis=0),size=args.batch_size)
        Y_mb = I_train[ind_train,:]
        X_mb = X_train[ind_train,:]
        
        feed_dict={x:X_mb,y:Y_mb}
        results=sess.run(fetches,feed_dict)
        Lxs,Lys,_,summary_md1=results
        if i%100==0:
            ind_test = np.random.randint(0,np.size(I_test,axis=0),size=args.batch_size)
            Y_mb_test = I_test[ind_test,:]
            X_mb_test = X_test[ind_test,:]
            feed_dict_test={x:X_mb_test,y:Y_mb_test}
            Lxs,Lys,summary_test=sess.run(fetches_test,feed_dict_test)
            print("iter=%d : MSE: %f fidelity: %f" % (i,Lxs,Lys))
        model1_writer.add_summary(summary_md1, i)
        model2_writer.add_summary(summary_test, i)
    #save trained weights
    save_path=saver.save(sess,save_model_path)

#%% generate and save reconstruction examples
n_instance=1
n_samples=1000
# feed test data into retrieval network
for kk in range(n_instance):
    samples_test = sess.run(Xp,feed_dict={y:I_test})
    MSE_test=np.mean(np.mean(np.mean((samples_test-X_test[0:n_samples,:])**2,axis=1)))
    PSNR_test=10*np.log10(1/MSE_test)
    AX_array=sess.run(A_fun(samples_test))
    fidelity_test=np.mean(np.mean(np.mean((AX_array-I_test[0:n_samples,:])**2,axis=1)))
    savemat(save_mat_folder+'{}_test_branch2.mat'.format(str(kk)),\
            {'sample_test':samples_test,'AX_array':AX_array,'MSE_test':MSE_test,'fidelity_test':fidelity_test})