
# coding: utf-8

# In[1]:

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import csv
import pandas as pd
from __future__ import print_function
from keras.utils import np_utils
from keras.utils import plot_model
#plot model
import pydot
pydot.find_graphviz = lambda: True


# In[2]:

#load train data
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
x_train = x_train.reshape(28709,48*48).astype('float32')
x_labels = x_labels.apply(pd.to_numeric).as_matrix()


# In[13]:

#load test data
x_test = pd.read_csv("test.csv", encoding = "big5", low_memory="false", header = None)
x_test = x_test[1:]
x_test = x_test[:][1]
x_testcon = []
for i in x_test:
    temp = i.split()
    k = list(map(int, temp))
    x_testcon.append(np.array(k))
x_test = np.array(x_testcon)
x_test = x_test.reshape(7178,48*48).astype('float32')


# In[4]:

# adjust the datas
y_train = np_utils.to_categorical(x_labels, 7)
x_train = x_train/255
x_test = x_test/255


# In[9]:

act = keras.optimizers.rmsprop(lr=0.0003, decay=1e-6)
model2 = Sequential()
model2.add(Dense(units=1288,activation='relu',input_shape=(48*48,)))
model2.add(BatchNormalization())
model2.add(Dropout(0.3))
model2.add(Dense(units=1024,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.2))
model2.add(Dense(units=784,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.2))
model2.add(Dense(units=666,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.2))
model2.add(Dense(units=512,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.2))
model2.add(Dense(units=512,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.2))
model2.add(Dense(units=384,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.2))
model2.add(Dense(units=256,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.2))
model2.add(Dense(units=256,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.3))
model2.add(Dense(units=128,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.3))
model2.add(Dense(units=64,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.3))
model2.add(Dense(units=32,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.3))
model2.add(Dense(units=16,activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.3))
model2.add(Dense(units=7,activation='softmax'))
model2.summary()
model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
plot_model(model2, to_file='model_dnn2.png')


# In[10]:

# datagenerator = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=False)  # randomly flip images

# datagenerator.fit(x_train)
# history = model2.fit_generator(datagenerator.flow(x_train, y_train, batch_size=128),
#                         steps_per_epoch=x_train.shape[0]/128,
#                         epochs=22)
history = model2.fit(x_train,y_train,validation_split=0.01,batch_size=128,epochs=50)


# In[11]:

score = model2.evaluate(x_train,y_train)
print('\nTrain Acc:', score[1])


# In[14]:

predictions = model2.predict_classes(x_test)
print(predictions)


# In[15]:

f = open('out_dnn.csv', 'w')
f.write("id,label\n")
for i in range(0,7178) :
    tmp = str(i) + "," + str(predictions[i]) + "\n"  
    f.write(tmp)
f.close()
print('done') 


# In[16]:

#plot
# import matplotlib.pyplot as plt
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# In[17]:

history = model2.fit(x_train,y_train,validation_split=0.01,batch_size=128,epochs=50)


# In[ ]:

score = model2.evaluate(x_train,y_train)
print('\nTrain Acc:', score[1])
predictions = model2.predict_classes(x_test)


# In[ ]:

f = open('out_25.csv', 'w')
f.write("id,label\n")
for i in range(0,7178) :
    tmp = str(i) + "," + str(predictions[i]) + "\n"  
    f.write(tmp)
f.close()
print('done') 

