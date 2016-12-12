
# coding: utf-8

# In[373]:

import os
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.convolutional import AveragePooling2D, Convolution2D, MaxPooling2D, ZeroPadding2D  
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
from keras.utils import np_utils

from tempfile import TemporaryFile
import PIL
import random

#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

# In[35]:

pathToImages = "/Users/bsoper/Dropbox/Senior Year/Fall/Machine Learning/Project/liscencePlateClassification/Characters/"
scaledImageSize = (100,100)

#Organize Data so that it is in numpy arrays for the neural network
def changImageSize(fileName):
    img = PIL.Image.open(fileName) #Opens as a color image
    #img = mpimg.imread(pathToImages+fileName) #Opens as a black and white image
    img = img.convert('L') #Converts color image to greyscale
    img = img.resize(scaledImageSize, PIL.Image.ANTIALIAS)
    return img


# In[260]:

imageData = np.expand_dims(np.empty([100,100]), axis=0)
locationData = np.ndarray(shape=(1,5795), dtype=float) #np.empty([1,1]) # 5795
i = 0
for root, dirs, files in os.walk(pathToImages):
    #print os.path.basename(root) 
    for fileName in files:
        if fileName == '.DS_Store': continue
        #if i >= 300: break
        img = changImageSize(root + '/' + fileName)
        imageData = np.concatenate((imageData, np.expand_dims(np.asarray(img),axis=0)), axis = 0)
        locationData[0,i] = os.path.basename(root) # = np.concatenate((locationData,np.expand_dims(np.asarray([os.path.basename(root)]), axis = 0)), axis = 0)
        i += 1
        
imageData = imageData[1:]           #These two lines get ride of the garbage that was created in the empty array with np.empty() above
locationData = locationData[0,:]

# In[262]:

#normalize data
imageData = imageData/255

# In[302]:

# shuffle the order samples
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


# In[340]:

# shuffle data
imageData, locationData = shuffle_in_unison(imageData, locationData)

#divide Data into test and train
data = imageData #np.expand_dims(imageData, axis=1)
X_train = data[:5000]
X_test = data[5000:]
y_train = locationData[:5000]
y_test = locationData[5000:]

# In[344]:

num_train = X_train.shape[0]
num_test = X_test.shape[0]
im_width  = X_train.shape[1]
im_height = X_train.shape[2]

# change type to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[346]:

# convert class vectors to binary class matrices (one hot representation)
nb_classes = np.unique(y_train).size
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)
X_train.shape


# In[374]:

model = Sequential()
model.add(Convolution2D(64, 10, 10, border_mode='same', input_shape=(1,im_width,im_height)))

model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(96, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(96, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 2, 2, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 2, 2, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, W_regularizer=l2(1e-3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()


# In[375]:

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


# In[377]:

batch_size = 100
nb_epoch = 50
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))


# In[283]:

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])



