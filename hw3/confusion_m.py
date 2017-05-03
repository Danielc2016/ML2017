
# coding: utf-8

# In[1]:

import keras
from keras.models import Sequential
import numpy as np
import csv
import pandas as pd
from __future__ import print_function
from keras.utils import np_utils
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from keras.models import load_model
# from marcos import exp_dir
import matplotlib.pyplot as plt
import itertools
import pydot
pydot.find_graphviz = lambda: True


# In[8]:

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Predicted')
    plt.xlabel('True Label')


# In[4]:

emotion_classifier = load_model('near2best.h5')


# In[5]:

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
x_train = x_train/255


# In[6]:

predictions = emotion_classifier.predict_classes(x_train)


# In[9]:

df_confusion = confusion_matrix(x_labels,predictions)
plt.figure()
plot_confusion_matrix(df_confusion, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.show()


# In[ ]:



