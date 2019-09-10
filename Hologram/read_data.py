import numpy as np
import hdf5storage
def read_data(train_dataset='./Hologram/dataset/hologram_MNIST_64X64_train_Aref=0.mat',test_dataset='./Hologram/dataset/hologram_MNIST_64X64_test_Aref=0.mat'):
    load=hdf5storage.loadmat(train_dataset)
    f_train=np.float32(load['f_train'])
    f_train=np.transpose(f_train,(2,0,1))
    f_train=np.expand_dims(f_train,3)
    I_train=np.float32(load['I_train'])
    I_train=np.transpose(I_train,(2,0,1))
    I_train=np.expand_dims(I_train,3)
    load=hdf5storage.loadmat(test_dataset)
    f_test=np.float32(load['f_test'])
    f_test=np.transpose(f_test,(2,0,1))
    f_test=np.expand_dims(f_test,3)
    I_test=np.float32(load['I_test'])
    I_test=np.transpose(I_test,(2,0,1))
    I_test=np.expand_dims(I_test,3)
    return f_train, I_train, f_test, I_test