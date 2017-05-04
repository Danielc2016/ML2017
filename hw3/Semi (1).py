
# coding: utf-8

# In[10]:

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import csv
import pandas as pd
from keras.utils import np_utils


# In[2]:

x_in = pd.read_csv("train.csv", encoding = "big5", low_memory="false", header = None)
x_in = x_in[1:]
x_labels = x_in[:][0]
x_images = x_in[:][1]
# x_train = x_images.reshape(28709, 48*48)
x_train = []
for i in x_images:
    temp = i.split()
    k = list(map(int, temp))
    x_train.append(np.array(k))
x_train = np.array(x_train)
x_train = x_train.reshape(28709,48,48,1).astype('float32')
x_labels = x_labels.apply(pd.to_numeric).as_matrix()


# In[3]:

y_train = np_utils.to_categorical(x_labels, 7)
x_train = x_train/255


# In[7]:

input_img = Input(shape=(48,48,1))
x = Conv2D(32,(3,3), padding='same', activation='relu')(input_img)
x = Conv2D(32,(3,3), padding='same', activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(16,(3,3), padding='same', activation='relu')(input_img)
x = Conv2D(16,(3,3), padding='same', activation='relu')(x)
x = MaxPooling2D((2,2))(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
# encoder create
x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


# In[11]:

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[16]:

autoencoder.fit(X_unlabel, X_unlabel,
                nb_epoch=10,
                batch_size=128,
                shuffle=True,
                validation_data=(X_train, X_train))


# In[ ]:



