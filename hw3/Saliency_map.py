
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras import backend as K

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_saliency
import cv2


# In[2]:

model2 = load_model('near2best.h5')


# In[3]:

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
x_train = x_train.reshape(28709,48,48,1).astype('float32')
x_labels = x_labels.apply(pd.to_numeric).as_matrix()


# In[4]:

layer_idx = [idx for idx, layer in enumerate(model2.layers) if layer.name == 'dense_3'][0]


# In[5]:

heatmaps = []
c = []
c = model2.predict_classes(x_train[0:100])
# def compile_saliency_function(model):
#     """
#     Compiles a function to compute the saliency maps and predicted classes
#     for a given minibatch of input images.
#     """
#     inp = model.layers[0].get_input()
#     outp = model.layers[-1].get_output()
#     max_outp = T.max(outp, axis=1)
#     saliency = theano.grad(max_outp.sum(), wrt=inp)
#     max_class = T.argmax(outp, axis=1)
#     return theano.function([inp], [saliency, max_class])


# In[7]:

# Show the heatmap
for i in range(10):
    heatmap = visualize_saliency(model2, layer_idx, c[i+10], x_train[i+10], alpha=0.7)
    # img = visualize_saliency(model2, layer_idx, c[9], x_train[9], alpha=0)
    heatmaps.append(heatmap)


# In[8]:

# plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Saliency map')
plt.show()


# In[ ]:

# img = visualize_saliency(model2, layer_idx, c[9], x_train[9], alpha=0.001)


# In[ ]:

# a = img-heatmap


# In[ ]:




# In[ ]:



