#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
encoder/decoder structures for hologram

@author: zyzhu
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Conv2DTranspose
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2DCell
from aux_layers import sampler_conv

def encoderY(args):
    with tf.name_scope('encoderY'):
        y=tf.keras.Input(shape=(args.N_det,args.N_det,1),name='Input_encoderY')
        I_conv=Conv2D(4,(128,128),(1,1),padding='same')(y)
        I_conv=LeakyReLU()(I_conv)
        I_conv=Conv2D(args.enc_size,(64,64),(1,1),padding='same')(I_conv)
        I_conv=LeakyReLU()(I_conv)
        encodery_model=tf.keras.Model(inputs=y,outputs=I_conv,name='encoderY')
        return encodery_model

def encoderX(args):
    with tf.name_scope('encoderX'):
        x=tf.keras.Input(shape=(args.N_det,args.N_det,1),name='Input_encoderX')
        x_temp=Conv2D(args.enc_size,(128,128),(1,1),padding='same')(x)
        x_temp=LeakyReLU()(x_temp)
        encoderx_model=tf.keras.Model(inputs=x,outputs=x_temp,name='encoderX')
        return encoderx_model

def decoder(args):
    with tf.name_scope('decoder'):
        y=tf.keras.Input(shape=(args.N_det,args.N_det,args.dec_size),name='Input_decoder')
        I_out=Conv2DTranspose(4,(64,64),(1,1),padding='same')(y)
        I_out=Conv2DTranspose(1,(128,128),(1,1),padding='same')(y)
        decoder_model=tf.keras.Model(inputs=y,outputs=I_out,name='decoder')
        return decoder_model

#%% building-block models: recogntion, conditional prior and generator
class recognition_encoder(tf.keras.Model):
    def __init__(self, args):
        super(recognition_encoder,self).__init__(name='recog_enc')
        with tf.name_scope('recog_enc'):
            self.encodeY = encoderY(args)
            self.encodeX = encoderX(args)
        
    def call(self,y,x):
        ry = self.encodeY(y)
        rx = self.encodeX(x)
        ryx= tf.concat((ry,rx),axis=-1)
        return ryx

class recognition_model(tf.keras.Model):
    def __init__(self, args):
        super(recognition_model,self).__init__(name='recognition')
        with tf.name_scope('recognition'):
            self.encodeY = encoderY(args)
            self.encodeX = encoderX(args)
            self.LSTM_enc_cell = ConvLSTM2DCell(filters=args.enc_size,\
                                                kernel_size=args.convlstm_kernel_size,\
                                                strides=1,\
                                                padding='same',\
                                                name='LSTM_encoder')
            self.sample=sampler_conv(args.enc_size,args.z_size,args.N_obj,args.N_obj)
            #add LSTM_enc_cell to graph with a dummy call to itself
            h=tf.keras.Input(shape=(args.N_obj,args.N_obj,12))
            state=[tf.keras.Input(batch_size=args.batch_size,shape=(args.N_obj,args.N_obj,args.enc_size),dtype='float32'),\
                   tf.keras.Input(batch_size=args.batch_size,shape=(args.N_obj,args.N_obj,args.enc_size),dtype='float32')]
            self.LSTM_enc_cell(inputs=h,states=state)
        
    def call(self,y,x,state):
        ry = self.encodeY(y)
        rx = self.encodeX(x)
        ryx= tf.concat((ry,rx),axis=-1)
        h_enc,state_new = self.LSTM_enc_cell(inputs=ryx,states=state)
        z,mu,logsigma,sigma = self.sample(h_enc)
        dist_params = [mu,logsigma,sigma]
        return z,dist_params,ry,state_new
    
class cond_prior_model(tf.keras.Model):
    def __init__(self, args):
        super(cond_prior_model,self).__init__(name='cond_prior')
        with tf.name_scope('cond_prior'):
            self.encodeY = encoderY(args)
            self.LSTM_enc_cell = ConvLSTM2DCell(filters=args.enc_size,\
                                                kernel_size=args.convlstm_kernel_size,\
                                                strides=1,\
                                                padding='same',\
                                                name='LSTM_encoder')
            self.sample=sampler_conv(args.enc_size,args.z_size,args.N_obj,args.N_obj)
            #add LSTM_enc_cell to graph with a dummy call to itself
            h=tf.keras.Input(shape=(args.N_obj,args.N_obj,args.enc_size))
            state=[tf.keras.Input(batch_size=args.batch_size,shape=(args.N_obj,args.N_obj,args.enc_size),dtype='float32'),\
                   tf.keras.Input(batch_size=args.batch_size,shape=(args.N_obj,args.N_obj,args.enc_size),dtype='float32')]
            self.LSTM_enc_cell(inputs=h,states=state)
        
    def call(self,y,state):
        ry = self.encodeY(y)
        h_enc,state_new = self.LSTM_enc_cell(inputs=ry,states=state)
        z,mu,logsigma,sigma = self.sample(h_enc)
        dist_params = [mu,logsigma,sigma]
        return z,dist_params,ry,state_new

class generator_model(tf.keras.Model):
    def __init__(self, args):
        super(generator_model,self).__init__(name='generator')
        with tf.name_scope('generator'):
            self.decode = decoder(args)
            self.LSTM_dec_cell = ConvLSTM2DCell(filters=args.dec_size,\
                                                kernel_size=args.convlstm_kernel_size,\
                                                strides=1,\
                                                padding='same',\
                                                name='LSTM_decoder')
            #add LSTM_enc_cell to graphy with a dummy call to itself
            h=tf.keras.Input(shape=(args.N_obj,args.N_obj,args.z_size+args.enc_size))
            state=[tf.keras.Input(batch_size=args.batch_size,shape=(args.N_obj,args.N_obj,args.dec_size),dtype='float32'),\
                   tf.keras.Input(batch_size=args.batch_size,shape=(args.N_obj,args.N_obj,args.dec_size),dtype='float32')]
            self.LSTM_dec_cell(inputs=h,states=state)
        
    def call(self,ry,z,state):
        ryz = tf.concat((ry,z),axis=-1)
        h_dec,state_new = self.LSTM_dec_cell(inputs=ryz,states=state)
        x = self.decode(h_dec)
        return x, state_new