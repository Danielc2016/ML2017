
# coding: utf-8

# In[ ]:

import word2vec
import numpy as np
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text


# In[ ]:

# concatenate the 7 files
# import codecs
# filenames = ['t1.txt', 't2.txt', 't3.txt', 't4.txt', 't5.txt', 't6.txt', 't7.txt']
# with open('result.txt', 'w') as outfile:
#     for fname in filenames:
#         with open(fname) as infile:
#             for line in infile:
#                 outfile.write(line)


# In[ ]:

# train model
MIN_COUNT = 5
MODEL = 1

word2vec.word2vec(train='result.txt',
        output='word_model.bin',
        threads=4,
        window=12,
        min_count=MIN_COUNT,
        verbose=True)


# In[ ]:

# load model
model2 = word2vec.load('word_model.bin')
vocabs = []                 
vecs = []                   
for vocab in model2.vocab:
    vocabs.append(vocab)
    vecs.append(model2[vocab])
vecs = np.array(vecs)[:820]
vocabs = vocabs[:820]


# In[ ]:

# Dimensionality Reduction
tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(vecs)


# In[ ]:

# filtering
use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]
    
plt.figure(figsize=(12, 8))
texts = []
for i, label in enumerate(vocabs):
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
             and all(c not in label for c in puncts)):
        x, y = reduced[i, :]
        texts.append(plt.text(x, y, label, color='blue'))
        plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='r', lw=0.5))

plt.savefig('w2v_820.png', dpi=600)
plt.show()


# In[ ]:



