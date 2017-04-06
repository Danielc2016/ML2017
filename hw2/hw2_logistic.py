
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import csv
from numpy import linalg
import sys
# hw2 logistic regression with adjusted features
raw1 = sys.argv[1]
raw2 = sys.argv[2]
trainxcsv = sys.argv[3]
trainycsv = sys.argv[4]
testcsv = sys.argv[5]
outcsv = sys.argv[6]


# In[2]:

x_in = pd.read_csv(trainxcsv, encoding = "big5", low_memory="false", header = None)
x_in = x_in[1:]
x_train = x_in.apply(pd.to_numeric).as_matrix()
#x_in.reset_index(drop=True, inplace=True)


# In[3]:

y_in = pd.read_csv(trainycsv, encoding = "big5", low_memory="false", header = None)
y_train = y_in.apply(pd.to_numeric).as_matrix()
#pd.set_option("display.max_rows", 899)


# In[4]:

# adjusting interested features
for i in range(x_train.shape[0]):
    if x_train[i,3] < 7298:
        x_train[i,3] =0
    elif 30000<x_train[i,3]<42000:
        x_train[i,3] =0
    else:
        x_train[i,3] =2

for i in range(x_train.shape[0]):
    if 1810<x_train[i,4]<2000:
        x_train[i,4] = 1
    elif 2250<x_train[i,4]<2450:
        x_train[i,4] = 1
    else:
        x_train[i,4] =0.1
        
for i in range(x_train.shape[0]):
    if 25<x_train[i,0]<65:
        x_train[i,0] = 1
    else:
        x_train[i,0] = 0.2


# In[5]:

def sigmoidf(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999)


# In[6]:

def featurescale(x):
    import numpy as np
    x = np.matrix(x)
    #normedx = np.matrix(x, dtype=float)
    meanx = np.zeros(x.shape[1])
    stdx = np.zeros(x.shape[1])
    meanx = np.mean(x, axis=0)
    stdx = np.std(x, axis=0)
    normedx = (x - meanx)/stdx
#     for i in range(x.shape[1]):
#         meanx[i] = np.mean(x[:,i])
#         stdx = np.std(x[:,i])
#         normedx[:, i] = (x[:, i] - meanx[i])/stdx[i]
    return normedx


# In[7]:

lr = 0.035
w = np.zeros(shape = (x_train.shape[1],1))
rounds = 9000
londaa = 0.1
w[3] = 100
w[4] = 2


# In[8]:

x_train = featurescale(x_train)


# In[9]:

def gradientdescent(x, y, weight, l_rate, iters, londa):
    x = np.matrix(x)
    y = np.matrix(y)
    weight = np.matrix(weight)
    m = x.shape[0]
    ada = np.zeros(weight.shape)
    for i in range(iters):
        temp = x.T * (sigmoidf(x.dot(weight))-y)
        temp += londa * w
        temp = (l_rate/m)*temp
        ada += np.square(temp)
        temp = temp/np.sqrt(ada)
        weight -= temp
        if i%500 == 0:
            print(weight[3])
    return weight


# In[10]:

w_test = np.asarray(gradientdescent(x_train, y_train, w, lr, rounds, londaa))


# In[11]:

def cross_entropy(x, y, weight, londa):
    temp = sigmoidf(x.dot(weight))
    y = np.matrix(y)
    m = y.shape[0]
    y_prime = np.ones(shape=(y.shape))-y
    x_prime = np.ones(shape=(temp.shape))-temp
    regu = londa*(np.sum(np.square(w)))/2
    return -((y.T*np.log(temp))+(y_prime.T*np.log(x_prime))+regu)/m


# In[12]:

cross_entropy(x_train, y_train, w_test, londaa)


# In[13]:

x_test = pd.read_csv(testcsv, encoding = "big5", low_memory="false", header = None)
x_test = x_test[1:]


# In[14]:

x_test = x_test.apply(pd.to_numeric).as_matrix()


# In[15]:

for i in range(x_test.shape[0]):
    if x_test[i,3] < 7298:
        x_test[i,3] =0.1
    elif 30000<x_test[i,3]<42000:
        x_test[i,3] =0.1
    else:
        x_test[i,3] =2


for i in range(x_test.shape[0]):
    if 1810<x_test[i,4]<2000:
        x_test[i,4] = 2
    elif 2250<x_test[i,4]<2450:
        x_test[i,4] =2
    else:
        x_test[i,4] =0.1        
        
for i in range(x_test.shape[0]):
    if 25<x_test[i,0]<65:
        x_test[i,0] = 1.5
    else:
        x_test[i,0] = 0.1


# In[16]:

def predict(x, weight):
    x_scale = featurescale(x)
    np.place(x_scale, np.isnan(x_scale),0)
    temp = sigmoidf(x_scale.dot(weight))
    y_out = np.around(temp)
    with open(outcsv, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_out):
            f.write('%d,%d\n' %(i+1, v))
    print('done')
    return


# In[17]:

predict(x_test, w_test)


# In[ ]:



