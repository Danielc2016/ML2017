
# coding: utf-8

# In[1]:

"""
Created on Mon Mar 20 17:22:05 2017
@author: Daniel
"""
import csv 
import numpy as np
import sys

traincsv = sys.argv[1]
testcsv = sys.argv[2]
outcsv = sys.argv[3]

# In[2]:

Data = []
for i in range(18):
    Data.append([])

n_row = 0
text = open(traincsv, 'r', encoding = "ISO-8859-1") 
row = csv.reader(text , delimiter=",")
for r in row:
    if n_row != 0:
        for i in range(3,27):
            if r[i] != "NR":
                Data[(n_row-1)%18].append( float( r[i] ) )
            else:
                Data[(n_row-1)%18].append( float( 0 ) ) 
    n_row =n_row+1
text.close()


# In[3]:

train_x = []
train_y = []


# In[4]:

for i in range(12):
    for j in range(471):
        train_x.append( [1] )
        for t in range(18):
            for s in range(9):
                train_x[471*i+j].append( Data[t][480*i+j+s] )
        train_y.append( Data[9][480*i+j+9] )

train_x = np.array(train_x)
train_y = np.array(train_y)
b = 0 # initial b
w = np.zeros(163) # initial weight with 0
w[162] = 0.05
w[155] = 0.05
w[144] = 0.05
lr = 0.104 # learning rate
iteration = 55000


# In[5]:

b_lr = 0.0
w_lr = np.zeros(163)

##  print(np.shape(train_y))
prev_grad = np.zeros(163)
prev_grad[162] = 0.05
prev_grad[155] = 0.05
prev_grad[144] = 0.05
b_grad = 0.0
# add regularization
lamda = 100
for i in range(iteration):
    
    #for n in range(len(train_x)):        
    # b_grad = b_grad  - 2.0*(train_y  - np.dot(train_x[n],w))*1.0
    #y_guess = np.zeros((1, 163), dtype=object)
    grad = np.dot(np.transpose(train_x),(np.dot(train_x,w)-train_y)) *2
    prev_grad += (grad**2 + lamda*(w**2))
    ada = np.sqrt(prev_grad)
    w -= lr*grad/ada
    #print(np.sqrt(np.mean(np.square(np.dot(train_x,w)-train_y))))
    #print('0')
#print(w)


# In[6]:

test_x = []
test = open(testcsv, 'r', encoding = "ISO-8859-1") 
rows = csv.reader(test , delimiter=",")
m_row = 0
for r in rows:
    if (m_row%18==0):
        test_x.append(float(0.1))
    for i in range(2,11):
        if r[i] != "NR":
            test_x.append( float( r[i] ) )
        else:
            test_x.append( float( 0 ) )
    m_row +=1
test.close()


# In[7]:

np.shape(test_x)
test_x = np.reshape(test_x,(240,163))
test_y = np.dot(test_x, w)

f = open(outcsv, 'w')
f.write("id,value\n")
for i in range(0,240) :
    tmp = "id_" + str(i) + "," + str(test_y[i]) + "\n"  
    f.write(tmp)
f.close()
print('done')


# In[8]:

print(np.sqrt(np.mean(np.square(np.dot(train_x,w)-train_y))))


# In[ ]:



