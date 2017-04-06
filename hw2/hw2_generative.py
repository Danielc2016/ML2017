
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import csv
import sys
#hw2 probalistic generative model


# In[2]:

from numpy import linalg
np.set_printoptions(suppress=True)
raw1 = sys.argv[1]
raw2 = sys.argv[2]
trainxcsv = sys.argv[3]
trainycsv = sys.argv[4]
testcsv = sys.argv[5]
outcsv = sys.argv[6]


# In[3]:

x_in = pd.read_csv(trainxcsv, encoding = "ISO-8859-1", low_memory="false")
x_in = x_in[1:]
x_train = x_in.apply(pd.to_numeric).as_matrix()


# In[4]:

y_in = pd.read_csv(trainycsv, encoding = "ISO-8859-1")
y_train = y_in.apply(pd.to_numeric).as_matrix()


# In[5]:

# putting x_trains into two classes
n1 = 0
n2 = 0
class1 = []
class2 = []


# In[6]:

row = 0
for i in y_train[:,0]:
    if i ==0 :
        n1+=1
        class1.append(x_train[row, 0:])
    else:
        n2+=1
        class2.append(x_train[row, 0:])
    row += 1


# In[7]:

class1 = np.asarray(class1).reshape(n1, x_train.shape[1])
class2 = np.asarray(class2).reshape(n2, x_train.shape[1])
class2.shape


# In[8]:

cov1 = np.cov(class1.T)
cov2 = np.cov(class2.T)
finalcov = np.matrix(cov1*(n1/(n1+n2)) + cov2*(n2/(n1+n2)))
inverse_cov = linalg.pinv(finalcov)


# In[9]:

mu1 = []
mu2 = []

for i in range(x_train.shape[1]):
    mu1.append(np.mean(class1[:,i]))
    mu2.append(np.mean(class2[:,i]))

mu1 = np.asarray(mu1).reshape(x_train.shape[1],1)
mu2 = np.asarray(mu2).reshape(x_train.shape[1],1)
# inverse_cov = linalg.pinv(finalcov) 


# In[10]:

w = (mu1 - mu2).T.dot(inverse_cov)

b = -0.5 * mu1.T.dot(inverse_cov).dot(mu1) + 0.5 * mu2.T.dot(inverse_cov).dot(mu2) + np.log(n1/n2)
# b = - 0.5*mu1.T*inverse_cov.dot(mu1) + 0.5*mu2.T*inverse_cov.dot(mu2) + np.log(n1/n2)


# In[11]:

x_testin = pd.read_csv(testcsv, encoding = "ISO-8859-1", low_memory="false")
x_test = x_testin.apply(pd.to_numeric).as_matrix()


# In[12]:

x_test.shape


# In[13]:

result = 1 - 1 / ( 1 + np.exp( - w.dot(x_test.T) - b ))


# In[14]:

r = []
result = np.asarray(result)
r.append(result[0])


# In[15]:

out = []
for i in range(0,16281):
    out.append(i+1)
    if result[0,i] >=0.46:
        out.append(1)
    else:
        out.append(0)
out = np.asarray(out).reshape(16281, 2)


# In[16]:

f = open(outcsv, 'w')
f.write("id,label\n")
for i in range(0,16281) :
    tmp = str(i+1) + "," + str(out[i][1]) + "\n"  
    f.write(tmp)
f.close()
print('done')


# In[17]:

# f = open("result.csv", 'w')
# f.write("id,value\n")
# for i in range(len(r)):
#     ts = str(r[i])
#     f.write(ts)
# f.close


# In[ ]:




# In[ ]:




# In[ ]:



