
# coding: utf-8

# In[1]:

import numpy as np
from hub_toolbox.IntrinsicDim import intrinsic_dimension
import sys


in_file = sys.argv[1]
out_file = sys.argv[2]

in_data = np.load(in_file)
new_arr = []
result = []


# In[3]:

for i in range(200):
    x = in_data[str(i)][:5566]
#     5566 cannot die!!!
    new_arr.append(x)


# In[4]:

for x2 in new_arr:
    a = intrinsic_dimension(x2, k1=7, k2=15, estimator='levina', trafo=None)
    result.append(a)


# In[17]:

new_result = []


# In[19]:

for i in result:
    if(i > 16):
        new_result.append((60*i - 60*16 + 22*16 -i*16)/(22 - 16)) 
    else:
        new_result.append(i)



ln_result = np.log(new_result)


# In[21]:

print(ln_result)


# In[22]:

f = open(out_file, 'w')
f.write("SetId,LogDim\n")
for i in range(0,200) :
    tmp = str(i) + "," + str(ln_result[i]) + "\n"  
    f.write(tmp)
f.close()
print('done') 




