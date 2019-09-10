#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary layers for RcVAE

@author: zyzhu
"""
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Dense, Conv2D
from tensorflow.contrib.rnn import BasicLSTMCell, Conv2DLSTMCell

def KL_calc(mu_q,mu_p,logsigma_q,logsigma_p,sigma_q,sigma_p):
    with tf.name_scope('KL_calc'):
        T=len(mu_q)
        kl_terms=[0]*T
        for t in range(T):
            muq=mu_q[t]
            mup=mu_p[t]
            logsigmaq=logsigma_q[t]
            logsigmap=logsigma_p[t]
            sigmaq=sigma_q[t]
            sigmap=sigma_p[t]
            kl_terms[t]=tf.reduce_sum((tf.square(muq-mup)+tf.square(sigmaq))/(2*tf.square(sigmap))-(logsigmaq-logsigmap)-0.5,1)
        KL=tf.add_n(kl_terms) 
        Lz=tf.reduce_mean(KL,name='KL')
        return Lz

def split(x):
    return tf.split(x,num_or_size_splits=2,axis=-1)

def sampling(args):
    mu,sigma=args
    return mu+sigma*tf.random_normal(shape=tf.shape(mu), mean=0, stddev=1)

def sampler(enc_size,z_size,name='sampler'):
    with tf.name_scope(name):
        h_enc=tf.keras.Input(shape=(enc_size,))
        h_enc_mu,h_enc_sigma=Lambda(split,name='split')(h_enc)
        mu=Dense(z_size,kernel_initializer='glorot_normal',name='mu')(h_enc_mu)
        logsigma=Dense(z_size,kernel_initializer='glorot_normal',name='logsigma')(h_enc_sigma)
        sigma=Lambda(tf.exp,name='sigma')(logsigma)
        z=Lambda(sampling,name='z')([mu,sigma])
        sampler_model=tf.keras.Model(inputs=h_enc,outputs=[z,mu,logsigma,sigma],name=name)
        return sampler_model
    
def sampler_conv(enc_size,z_size,Nrow,Ncol,name='sampler'):
    with tf.name_scope(name):
        h_enc=tf.keras.Input(shape=(Nrow,Ncol,enc_size))
        h_enc_mu,h_enc_sigma=Lambda(split,name='split')(h_enc)
        mu=Conv2D(z_size,(1,1),(1,1),padding='same',kernel_initializer='glorot_normal',name='mu')(h_enc_mu)
        logsigma=Conv2D(z_size,(1,1),(1,1),padding='same',kernel_initializer='glorot_normal',name='logsigma')(h_enc_sigma)
        sigma=Lambda(tf.exp,name='sigma')(logsigma)
        z=Lambda(sampling,name='z')([mu,sigma])
        sampler_model=tf.keras.Model(inputs=h_enc,outputs=[z,mu,logsigma,sigma],name=name)
        return sampler_model
    
def LSTM_init_state(batch_size,output_size):
    with tf.name_scope('LSTM_init_state'):
        init_state=tf.zeros(dtype='float32',shape=(batch_size,2*output_size))
#        init_state=tf.zeros_like(init_state)
        init_state=tf.split(init_state,num_or_size_splits=2,axis=-1)
#        cell=LSTMCell(output_size)
#        init_state=cell.get_initial_state(batch_size=batch_size,dtype='float32')
        return init_state

def BasicLSTM_init_state(batch_size,output_size):
    with tf.name_scope('LSTM_init_state'):
#        init_state=tf.zeros(dtype='float32',shape=(batch_size,2*output_size))
#        init_state=tf.zeros_like(init_state)
#        init_state=tf.split(init_state,num_or_size_splits=2,axis=-1)
        cell=BasicLSTMCell(output_size)
        init_state=cell.zero_state(batch_size=batch_size,dtype='float32')
        return init_state
    
def ConvLSTM_init_state(batch_size,in_channel,out_channel,Nrow,Ncol,convlstm_kernel_size):
    with tf.name_scope('ConvLSTM_init_state'):
#        init_state=tf.zeros(dtype='float32',shape=(batch_size,Nrow,Ncol,2*output_size))
#        init_state=tf.zeros_like(init_state)
#        init_state=tf.split(init_state,num_or_size_splits=2,axis=-1)
        cell=Conv2DLSTMCell(input_shape=[Nrow,Ncol,in_channel], kernel_shape=[convlstm_kernel_size,convlstm_kernel_size],output_channels=out_channel)
        init_state=cell.zero_state(batch_size,'float32')
        return init_state