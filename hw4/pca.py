
# coding: utf-8

# In[1]:

import os
from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt
import string
import sys


# In[2]:

# data_dir_path = sys.argv[1]
data_dir_path = './face/'
output_dir_path = './face/out/'


# In[3]:

def read_all_face():
#     first ten subjects
#     first ten faces
    filearray = []
    for i in range(10):
        for j in range(10):
            file_name = string.ascii_uppercase[i] + '{:02d}'.format(j) + '.bmp'
            filearray.append(data_dir_path + file_name)

    X = np.array([np.array(Image.open(fname)).flatten() for fname in filearray])
    
    return X


# In[4]:

f = read_all_face()


# In[19]:

def pca(face):
#     cmap=pyplot.get_cmap('gray')
    avg = np.mean(face, axis=0)
    face = face - avg
    # do svd!!!
    u,s,v = np.linalg.svd(face, full_matrices=False)
    # plot the top 9 eigenfaces
    plt.figure()
    plt.suptitle('eigenfaces')    
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(v[i,:].reshape(64, 64), cmap='gray')
        plt.axis('off')
    plt.savefig(output_dir_path + 'top9_eigenface.jpg')
    


# In[11]:

def eigenface(face):
    avg = np.mean(face, 0)
    face = face - avg
    # do svd!!!
    u,s,v = np.linalg.svd(face, full_matrices=False)
    # no.2 original images vs reconstruct face
    v = v[:5, :]
    w = np.dot(face, v.T)
    reconstruct_faces = avg + np.dot(w, v)
    # plot the recovered faces
    plt.figure()
    plt.suptitle('reconstructed faces')
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(reconstruct_faces[i, :].reshape(64, 64), cmap='gray')
        plt.axis('off')
        plt.savefig(output_dir_path + 'reconstructed_faces_.png')


# In[17]:

def eigenface_rmse(face):
    ori_face = face
    avg = np.mean(face, axis=0)

    face = face - avg
    
    # project the 100 faces onto the top k eigenfaces,
    for k in range(1, 101):
        u, s, v = np.linalg.svd(face.T, full_matrices=False)
        u = u[:, :k]
        weights = np.dot(face, u)

        recovered_faces = avg + np.dot(weights, u.T)

        rmse = (np.sqrt(((ori_face - recovered_faces) ** 2).mean()) / 255.0)
    
        if (rmse < 0.01):
            print ('the smallest k less than 0.01 is ' + str(k) )
            break       


def plot_average_face(face):
    # average for all column
    avg = np.mean(face, axis=0)
    im = Image.fromarray(np.uint8(avg.reshape(64, 64)))
    im.save(output_dir_path + 'avg_face_.bmp')


# In[ ]:

if __name__ == '__main__':
    f = read_all_face()
    plot_average_face(f)
    pca(f)
    eigenface(f)
    eigenface_rmse(f)
    print('done')

