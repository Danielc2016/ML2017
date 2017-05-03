
# coding: utf-8

# In[1]:

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import csv
import sys
import pandas as pd
from keras.utils import np_utils
from keras.models import load_model


# In[2]:

test_data = sys.argv[1]
out_file = sys.argv[2]


# In[3]:  CNN with 10 layerd for facial expression classification

#load test data
x_test = pd.read_csv(test_data, encoding = "big5", low_memory="false", header = None)
x_test = x_test[1:]
x_test = x_test[:][1]
x_testcon = []
for i in x_test:
    temp = i.split()
    k = list(map(int, temp))
    x_testcon.append(np.array(k))
x_test = np.array(x_testcon)
x_test = x_test.reshape(7178,48,48,1).astype('float32')


# In[5]:

# adjust the datas
x_test = x_test/255


# In[6]:

model2 = load_model('second_best.h5')


# In[7]:

predictions = model2.predict_classes(x_test)


# In[8]:

f = open(out_file, 'w')
f.write("id,label\n")
for i in range(0,7178) :
    tmp = str(i) + "," + str(predictions[i]) + "\n"  
    f.write(tmp)
f.close()
print('done') 


