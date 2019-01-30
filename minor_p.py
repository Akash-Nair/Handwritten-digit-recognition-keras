from keras.datasets import mnist
from keras.utils import np_utils
from the_network import Network
import numpy as np
import sys
import pickle
import cv2
import h5py
import matplotlib.pyplot as plt

np.random.seed(123)  # for reproducibility

#load data
import gzip
f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = pickle.load(f)
else:
    data = pickle.load(f, encoding='bytes')
f.close()
(x_train,y_train),(x_test,y_test) = data

#pre-processing
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#pre-process labels
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)


#setting the network
model = Network.create(28,28,1,10)

#compile model
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])

#fit model on training
model.fit(x_train,y_train,batch_size=32,epochs=1,verbose=2)

#evaluation
score = model.evaluate(x_test,y_test,verbose=0)
print('Accuracy is: ',score[1])

#save model
model.save('saved_network1.h5')

