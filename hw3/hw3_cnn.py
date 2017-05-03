
# coding: utf-8

# In[ ]:

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import csv
import pandas as pd
import sys
from __future__ import print_function
from keras.utils import np_utils
# from keras.utils import plot_model
# for confusion matrix
# from sklearn.metrics import confusion_matrix
# from keras.models import load_model

traincsv = sys.argv[1]

# In[ ]:

#load train data
x_in = pd.read_csv(traincsv, encoding = "big5", low_memory="false", header = None)
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


# In[ ]:

#load test data


# In[ ]:

# adjust the datas
y_train = np_utils.to_categorical(x_labels, 7)
x_train = x_train/255


# In[ ]:

# PReLU = keras.layers.advanced_activations.PReLU(init='zero', weights=None)
model2 = Sequential()
# act = keras.layers.advanced_activations.PReLU(alpha_initializer="zero", weights=None)
model2.add(Conv2D(64,(3,3), padding='same', activation='relu', input_shape=(48,48,1)))
model2.add(BatchNormalization())
model2.add(Conv2D(64,(3,3), padding='same', activation='relu', input_shape=(48,48,1)))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2,2)))
model2.add(Conv2D(128,(3,3), padding='same', activation='relu', input_shape=(48,48,1)))
model2.add(BatchNormalization())
model2.add(Conv2D(128,(3,3), padding='same', activation='relu', input_shape=(48,48,1)))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.2))
model2.add(Conv2D(256,(3,3), padding='same', activation='relu', input_shape=(48,48,1)))
model2.add(BatchNormalization())
model2.add(Conv2D(256,(3,3), padding='same', activation='relu', input_shape=(48,48,1)))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.25))
model2.add(Conv2D(512,(3,3), padding='same', activation='relu', input_shape=(48,48,1)))
model2.add(BatchNormalization())
# model2.add(Conv2D(512,(3,3), padding='same', activation='relu', input_shape=(48,48,1)))
# model2.add(BatchNormalization())
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(units=1024))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Dropout(0.45))
model2.add(Dense(units=1024))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Dropout(0.45))
model2.add(Dense(units=7,activation='softmax'))
model2.summary()
model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# plot_model(model2, to_file='model2.png')


# In[ ]:

# history = model2.fit(x_train,y_train,validation_split=0.1,batch_size=128,epochs=26)
datagenerator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagenerator.fit(x_train)
history = model2.fit_generator(datagenerator.flow(x_train, y_train, batch_size=128),
                        steps_per_epoch=x_train.shape[0]/128,
                        epochs=28)


#plot
# import matplotlib.pyplot as plt
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# In[ ]:

score = model2.evaluate(x_train,y_train)
print('\nTrain Acc:', score[1])


model2.save('second_best.h5')


# In[ ]:



