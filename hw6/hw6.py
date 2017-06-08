
# coding: utf-8

# In[1]:

import math
import pandas as pd
import numpy as np
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense, Flatten
from keras.models import Sequential, Model, load_model
from keras.layers import Concatenate, Input, BatchNormalization, Dot, Add
import sys


# In[2]:

def MF_ta(n_users, n_items, dim):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vec,item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = Model([user_input, item_input], r_hat)
    model.compile(loss='mse', optimizer='adamax')
    return model


# In[3]:

data_dir = sys.argv[1]
out_file = sys.argv[2]
test_file = data_dir+'test.csv'
test_csv = pd.read_csv(test_file)
test = np.asarray(test_csv)


# In[4]:

model = MF_ta(6040, 3952, 250)
model.load_weights('mf.h5')


# In[5]:

predictions = model.predict([test[:,1],test[:,2]])
predictions = predictions.reshape(100336,)


# In[6]:

f = open(out_file, 'w')
f.write("TestDataID,Rating\n")
for i in range(0,100336) :
    tmp = str(i+1) + "," + str(predictions[i]) + "\n"  
    f.write(tmp)
f.close()
print('done') 


# In[ ]:



