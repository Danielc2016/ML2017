
# coding: utf-8

# In[1]:

import math
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import numpy as np
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.layers import Flatten, Concatenate, Input, BatchNormalization, Dot, Add
from keras.models import Sequential, load_model
import sys


# In[8]:

data_dir = sys.argv[1]
out_file = sys.argv[2]
test_file = data_dir+'test.csv'
test_csv = pd.read_csv(test_file)
test = np.asarray(test_csv)


# In[3]:

class CFModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Dropout(0.2))
        P.add(Reshape((k_factors,)))
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Dropout(0.2))
        Q.add(Reshape((k_factors,)))
        super(CFModel, self).__init__(**kwargs)
        self.add(Merge([P, Q], mode='dot', dot_axes=1))

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]


# In[5]:

model2 = CFModel(6040, 3952, 300)
model2.load_weights('best_3.h5')
# model2 = load_model('best_3.h5')


# In[6]:

predictions = model2.predict([test[:,1],test[:,2]])
predictions = predictions.reshape(100336,)


# In[7]:

f = open(out_file, 'w')
f.write("TestDataID,Rating\n")
for i in range(0,100336) :
    tmp = str(i+1) + "," + str(predictions[i]) + "\n"  
    f.write(tmp)
f.close()
print('done') 


# In[ ]:



