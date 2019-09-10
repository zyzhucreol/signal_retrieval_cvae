#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pulse retrieval example

@author: Zheyuan Zhu
"""
import numpy as np
from scipy.io import savemat
import tensorflow as tf
from aux_layers import KL_calc, sampler_conv
from Hologram.layers import recognition_encoder, encoderY, decoder#, LSTM_decoder
from tensorflow.contrib.rnn import Conv2DLSTMCell
from Hologram.forward_model import A_fun
from Hologram.read_data import read_data

import os
os.environ["CUDA_VISIBLE_DEVICES"]='1';

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
save_name='RcVAE_hologram_1021'
save_model_path='./models/'+save_name # path to save trained model
load_model_path='./models/RcVAE_hologram_1021' # path to load pre-trained weights, if load_model=True
save_mat_folder='./RcVAE_results/'+save_name+'_' # path to save reconstruction examples
log_path='/media/HDD1/zyz/log_RcVAE_keras/'+save_name # Tensorboard path to log training process

#%% build RcVAE graph
x = tf.placeholder(tf.float32,shape=(None,args.N_obj,args.N_obj,1),name='X')
y = tf.placeholder(tf.float32,shape=(None,args.N_det,args.N_det,1),name='Y')
batch_size_flexible=tf.shape(y)[0]

x_q,mu_q,logsigma_q,sigma_q=[0]*args.T,[0]*args.T,[0]*args.T,[0]*args.T
x_p,mu_p,logsigma_p,sigma_p=[0]*args.T,[0]*args.T,[0]*args.T,[0]*args.T

recog_encode=recognition_encoder(args)
recog_lstm=Conv2DLSTMCell(input_shape=[args.N_obj,args.N_obj,2*args.enc_size], kernel_shape=[args.convlstm_kernel_size,args.convlstm_kernel_size], output_channels=args.enc_size,name='recog_lstm')
recog_states=recog_lstm.zero_state(batch_size_flexible,'float32')
recog_sample=sampler_conv(args.enc_size,args.z_size,args.N_obj,args.N_obj,name='recog_sampler')

cond_prior_encode=encoderY(args)
cond_prior_lstm=Conv2DLSTMCell(input_shape=[args.N_obj,args.N_obj,args.enc_size], kernel_shape=[args.convlstm_kernel_size,args.convlstm_kernel_size], output_channels=args.enc_size,name='cond_prior_lstm')
cond_prior_states=cond_prior_lstm.zero_state(batch_size_flexible,'float32')
cond_prior_sample=sampler_conv(args.enc_size,args.z_size,args.N_obj,args.N_obj,name='cond_prior_sampler')

dec_lstm=Conv2DLSTMCell(input_shape=[args.N_obj,args.N_obj,args.z_size], kernel_shape=[args.convlstm_kernel_size,args.convlstm_kernel_size], output_channels=args.dec_size,name='dec_lstm')
LSTM_dec_state_q=dec_lstm.zero_state(batch_size_flexible,'float32')
LSTM_dec_state_p=dec_lstm.zero_state(batch_size_flexible,'float32')

decode=decoder(args)

for t in range(args.T):
    # inference model
    x_q_prev = tf.zeros(shape=(),name='xq_init') if t==0 else x_q[t-1]
    delta_x = x if t==0 else x_q_prev
    deltay_q = y if t==0 else y-A_fun(x_q_prev)
    ryx = recog_encode(deltay_q,delta_x)
    h_enc_q,recog_states = recog_lstm(ryx,recog_states,scope='recog_lstm')
    z_q,mu_q[t],logsigma_q[t],sigma_q[t] = recog_sample(h_enc_q)
    h_dec_q,LSTM_dec_state_q = dec_lstm(z_q,LSTM_dec_state_q)
    dxq = decode(h_dec_q)
    x_q[t] = tf.add(x_q_prev,dxq,name='add_dxq{}'.format(str(t+1)))
    # retrieval model
    x_p_prev = tf.zeros(shape=(),name='xp_init') if t==0 else x_p[t-1]
    deltay_p = y if t==0 else y-A_fun(x_p_prev)
    rdy_p = cond_prior_encode(deltay_p)
    h_enc_p,cond_prior_states = cond_prior_lstm(rdy_p,cond_prior_states,scope='cond_prior_lstm')
    z_p,mu_p[t],logsigma_p[t],sigma_p[t] = cond_prior_sample(h_enc_p)
    h_dec_p,LSTM_dec_state_p = dec_lstm(z_p,LSTM_dec_state_p)
    dxp = decode(h_dec_p)
    x_p[t] = tf.add(x_p_prev,dxp,name='add_dxp{}'.format(str(t+1)))
    
#%% Define loss function and optimizer
Xq=x_q[-1]
Xp=x_p[-1]
Y_hat=A_fun(Xp)
KL_loss=KL_calc(mu_q,mu_p,logsigma_q,logsigma_p,sigma_q,sigma_p)
Lx=tf.losses.mean_squared_error(labels=x,predictions=Xq,scope='MSEX')
Ly=tf.losses.mean_squared_error(labels=y,predictions=Y_hat,scope='MSEY')
L_hybrid=tf.add(tf.add(Lx,0.001*KL_loss,'L'),0.5*Ly,'L_hybrid')

# Tensorboard monitoring
tf.summary.scalar('MSE_X',Lx)
tf.summary.scalar('MSE_Y',Ly)
tf.summary.scalar('KL',KL_loss)
tf.summary.image('Xq',Xq)
tf.summary.image('Xp',Xp)
tf.summary.image('X_true',x)
tf.summary.image('Y_true',y)
tf.summary.image('Yp',Y_hat)
merged=tf.summary.merge_all()

# optimizer
with tf.name_scope('train'):
    optimizer1=tf.train.AdamOptimizer(args.learning_rate)
#    grads1=optimizer1.compute_gradients(L_hybrid)
#    for i,(g,v) in enumerate(grads1):
#        if g is not None:
#            grads1[i]=(tf.clip_by_norm(g,1.0),v)
#    train_op1=optimizer1.apply_gradients(grads1)
    train_op1=optimizer1.minimize(L_hybrid)

# Setup training iterations
fetches=[Lx,Ly,KL_loss,train_op1,merged]
fetches_test=[Lx,Ly,KL_loss,merged]

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
        Lxs,Lys,Lzs,_,summary_md1=results
        if i%100==0:
            ind_test = np.random.randint(0,np.size(I_test,axis=0),size=args.batch_size)
            Y_mb_test = I_test[ind_test,:]
            X_mb_test = X_test[ind_test,:]
            feed_dict_test={x:X_mb_test,y:Y_mb_test}
            Lxs,Lys,Lzs,summary_test=sess.run(fetches_test,feed_dict_test)
            print("iter=%d : MSE: %f fidelity %f KL: %f" % (i,Lxs,Lys,Lzs))
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