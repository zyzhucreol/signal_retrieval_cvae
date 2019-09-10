import tensorflow as tf
import numpy as np
# parameters for the hologram
wl=np.float32(635e-6)
z_holo=np.float32(400)
dx=np.float32(0.05)
Aref=np.float32(0.0)
N_obj=64
N_det=64
X_dim=N_obj*N_obj
Y_dim=N_det*N_det

# nonlinear forward model
def Fresnel_prop_conv(fr,fi,z_holo,wavelength,N_obj,N_det,dx):
    # Fresnel propagation of field f; all units in mm
    n_obj_padding=int(np.floor((N_det-N_obj)/2))
    # zero-pad object f to match detector size in case it is smaller than detector
    frpad=tf.pad(fr,[[n_obj_padding,n_obj_padding],[n_obj_padding,n_obj_padding]])
    fipad=tf.pad(fi,[[n_obj_padding,n_obj_padding],[n_obj_padding,n_obj_padding]])
    # extend convolution kernel to twice its size to ensure accurate calculation
    x_det=tf.constant(np.arange(-np.floor(N_det),np.floor(N_det),1)*dx,dtype='float32')
    y_det=x_det
    x_det=tf.reshape(x_det,[1,-1])
    y_det=tf.reshape(y_det,[-1,1])
    x_det_rep=tf.tile(x_det,[N_det*2,1])
    y_det_rep=tf.tile(y_det,[1,N_det*2])
    Qr=tf.cos(np.pi/wavelength/z_holo*(tf.square(x_det_rep)+tf.square(y_det_rep)))
    Qi=tf.sin(np.pi/wavelength/z_holo*(tf.square(x_det_rep)+tf.square(y_det_rep)))
    Qr=tf.reverse(Qr,[0,1])
    Qi=tf.reverse(Qi,[0,1])
    Qr=tf.reshape(Qr,[N_det*2,N_det*2,1,1])
    Qi=tf.reshape(Qi,[N_det*2,N_det*2,1,1])
    frpad=tf.reshape(frpad,[1,N_det,N_det,1])
    fipad=tf.reshape(fipad,[1,N_det,N_det,1])
    frQr=tf.nn.conv2d(frpad,Qr,strides=[1,1,1,1],padding='SAME')
    frQi=tf.nn.conv2d(frpad,Qi,strides=[1,1,1,1],padding='SAME')
    fiQr=tf.nn.conv2d(fipad,Qr,strides=[1,1,1,1],padding='SAME')
    fiQi=tf.nn.conv2d(fipad,Qi,strides=[1,1,1,1],padding='SAME')
    Er=(frQr-fiQi)/N_det
    Ei=(frQi+fiQr)/N_det
    Er=tf.reshape(Er,[N_det,N_det])
    Ei=tf.reshape(Ei,[N_det,N_det])
    return Er, Ei

def apply_A(fr):
    fi=tf.zeros([N_obj,N_obj])
    (Er,Ei)=Fresnel_prop_conv(fr,fi,z_holo,wl,N_obj,N_det,dx)
    Iout=tf.square(Er+Aref)+tf.square(Ei)
    return Iout

def backprop(I):
    fi=tf.zeros([N_det,N_det])
    (Er,Ei)=Fresnel_prop_conv(I-Aref**2,fi,-z_holo,wl,N_det,N_det,dx)
    Iout=tf.square(Er)+tf.square(Ei)
    return Iout

def A_fun(f_batch):
    with tf.name_scope('forward_prop_nonlinear'):
        fn=lambda X: apply_A(X)
        f_batch=tf.reshape(f_batch,[-1,N_obj,N_obj])
        I_batch=tf.map_fn(fn,f_batch,name='AX')
        I_batch=tf.reshape(I_batch,[-1,N_det,N_det,1])
    return I_batch

def AT_fun(f_batch):
    fn=lambda X: backprop(X)
    f_batch=tf.reshape(f_batch,[-1,N_det,N_det])
    I_batch=tf.map_fn(fn,f_batch,name='back-prop_linear')
    return I_batch

def A_mat(fr):
    fi=tf.zeros([N_obj,N_obj])
    (Er,_)=Fresnel_prop_conv(fr,fi,z_holo,wl,N_obj,N_det,dx)
    Ilinear=Aref**2+2*Aref*Er
    return Ilinear

def Alinear(f_batch):
    fn=lambda X: A_mat(X)
    f_batch=tf.reshape(f_batch,[-1,N_obj,N_obj])
    I_batch=tf.map_fn(fn,f_batch,name='forward_prop_linear')
    return I_batch