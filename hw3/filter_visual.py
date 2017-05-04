
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy import misc
from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_activation, get_num_filters


# In[2]:

from keras.models import load_model
from keras import backend as K
model = load_model('near2best.h5')
# figure out conv2d_4


# In[3]:

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


# In[6]:

layer_name = 'conv2d_2'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
filters = np.arange(get_num_filters(model.layers[layer_idx]))


# In[10]:

vis_images = [visualize_activation(model, layer_idx, filter_indices=[0], seed_img=x_train[5], text=None)]


# In[25]:

vis = []
for i in range(15):
    vis_images = visualize_activation(model, layer_idx, filter_indices=[0], seed_img=x_train[5], text=None)
    vis_images = vis_images.reshape((48,48))
    vis.append(vis_images)


# In[14]:

vis_images.shape


# In[ ]:

# stitched = utils.stitch_images(vis_images)    
# plt.axis('off')
# plt.imshow(stitched)
# plt.title(layer_name)
# plt.show()


# In[37]:

def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional


# In[38]:

figures = {'filter'+str(i): vis[i] for i in range(6)}
plot_figures(figures, 2, 3)
plt.show()


# In[23]:

input_img =np.reshape(x_train[5], (48,48)) 
plt.imshow(input_img, cmap=plt.get_cmap('gray'))
plt.show()


# In[ ]:



