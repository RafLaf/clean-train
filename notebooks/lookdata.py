#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from PIL import Image
import matplotlib as mpl
from tqdm import tqdm
from sklearn.manifold import TSNE
import umap 
from sklearn.metrics import silhouette_score , silhouette_samples
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# # Visualizing the space with UMAP
# 
# ### Load data

# In[2]:


def access_data(letter,shot):
    feat = torch.load('features'+letter+str(shot),map_location=torch.device('cpu'))
    classifier= torch.load('classifier'+letter,map_location=torch.device('cpu'))
    accuracy = torch.load('complete_class_accuracy'+letter+str(shot)+'shots',map_location=torch.device('cpu'))
    idx = torch.load('complete_class_accuracy'+letter+'idx'+str(shot)+'shots',map_location=torch.device('cpu'))
    return feat,classifier,accuracy,idx


# In[3]:


shot=5
letter='A'
feat,classifier,acc,idx = access_data(letter,shot)
print(acc.shape)
print(feat.shape)
print(classifier.shape)
print(idx.shape)


# In[4]:


shot=5
letter='B'
featB,classifierB,accB,idxB = access_data(letter,shot)
print(accB.shape)
print(featB.shape)
print(classifierB.shape)
print(idxB.shape)


# In[5]:


base_mean = feat[:64].mean(-2)
base_meanB = featB[:64].mean(-2)
print(base_mean.shape)


# In[ ]:





# In[6]:



def sphering(features):
    return features / torch.norm(features, p = 2, dim = 2, keepdim = True)

def centering(train_features, features):
    return features - train_features.reshape(-1, train_features.shape[2]).mean(dim = 0).unsqueeze(0).unsqueeze(0)
feat_processed = sphering(centering(sphering(feat)[:64],sphering(feat) )) 


# In[7]:


mean_feat = feat.mean(-2)
mean_featp = feat_processed.mean(-2)
plt.imshow(torch.cdist(mean_featp,mean_featp))
plt.colorbar()


# In[8]:


plt.imshow(torch.cdist(mean_featp[-20:],mean_featp[-20:]))
plt.colorbar()


# In[9]:


umapA=umap.UMAP().fit_transform(feat_processed[-2:].reshape(-1,640))
plt.plot(umapA[:600,0],umapA[:600,1],'.',label='class vase')
plt.plot(umapA[600:,0],umapA[600:,1],'.',label='class cake')
plt.legend()


# In[10]:


umapA=umap.UMAP().fit_transform(feat_processed[80:82].reshape(-1,640))
plt.plot(umapA[:600,0],umapA[:600,1],'.',label='Microscopic worm')
plt.plot(umapA[600:,0],umapA[600:,1],'.',label='Crab')
plt.xlabel('axis 1')
plt.ylabel('axis 2')
plt.title('UMAP representation of 2 classes')
plt.legend()


# In[ ]:


umapA=umap.UMAP().fit_transform(feat[80:82].reshape(-1,640))
plt.plot(umapA[:600,0],umapA[:600,1],'.')
plt.plot(umapA[600:,0],umapA[600:,1],'.')


# In[ ]:


umapA=umap.UMAP().fit_transform(feat_processed[82:84].reshape(-1,640))
plt.plot(umapA[:600,0],umapA[:600,1],'.', label='dog breed')
plt.plot(umapA[600:,0],umapA[600:,1],'.', label='another dog breed')
plt.xlabel('axis 1')
plt.ylabel('axis 2')
plt.title('UMAP representation of 2 classes')
plt.legend()


# In[ ]:


umapA=umap.UMAP().fit_transform(torch.cat((feat_processed[80],feat_processed[95])).reshape(-1,640))
plt.plot(umapA[:600,0],umapA[:600,1],'.')
plt.plot(umapA[600:,0],umapA[600:,1],'.')


# In[ ]:


umapA=umap.UMAP().fit_transform(torch.cat((feat_processed[86],feat_processed[96])).reshape(-1,640))
plt.plot(umapA[:600,0],umapA[:600,1],'.')
plt.plot(umapA[600:,0],umapA[600:,1],'.')


# In[11]:


filenametrain = '/home/r21lafar/Documents/dataset/miniimagenetimages/train.csv'
filenameval = '/home/r21lafar/Documents/dataset/miniimagenetimages/validation.csv'
filenametest = '/home/r21lafar/Documents/dataset/miniimagenetimages/test.csv'
directory = '/home/r21lafar/Documents/dataset/miniimagenetimages/images/'
def opencsv(filename):
    file = open(filename)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    print(header)
    rowstrain = []
    rows = []
    for row in csvreader:
        rows.append(row)
    return rows
test = opencsv(filenametest)
train = opencsv(filenametrain)
val = opencsv(filenameval)
def openimg(cl,title=''):
    if cl<80:
        src=train
    if cl>=80:
        src=test
        cl-=80
    if cl>64 and cl<80:
        scr = val
        cl-=64
    if type(cl)==int:
        plt.figure(figsize=(5,5))
        idx=int((cl+0.5)*600)+np.random.randint(-100,100)
        filename=src[idx][0]
        im = Image.open(directory +filename)
        plt.title(title)
        plt.imshow(np.array(im))


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
openimg(1)


# In[17]:


umapA=umap.UMAP().fit_transform(feat_processed[-2:].reshape(-1,640))


# In[18]:


plt.plot(umapA[:600,0],umapA[:600,1],'.',c = 'blue')
plt.plot(umapA[600:,0],umapA[600:,1],'.',c = 'red')
plt.plot(umapA[:600,0].mean(),umapA[:600,1].mean(),'*',c = 'blue',markersize = 30)
plt.plot(umapA[600:,0].mean(),umapA[600:,1].mean(),'*',c = 'red',markersize = 30)


# ## Indice de confusion

# In[19]:


def indice_conf(cl1,cl2):
    featcl1 = feat_processed[cl1]
    featcl2 = feat_processed[cl2]
    mean_cl1 = featcl1.mean(0)
    mean_cl2 = featcl2.mean(0)
    d11 = torch.cdist(featcl1,mean_cl1.unsqueeze(0))
    d12 = torch.cdist(featcl1,mean_cl2.unsqueeze(0))
    d21 = torch.cdist(featcl2,mean_cl1.unsqueeze(0))
    d22 = torch.cdist(featcl2,mean_cl2.unsqueeze(0))
    bad1 = d12<d11
    bad2 = d21<d22
    return bad1.sum()/600,bad2.sum()/600


# In[20]:


confusion_matrix = torch.zeros((100,100))
for i in tqdm(range(100)):
    for j in range(i,100):
        confusion_matrix[i,j] , confusion_matrix[j,i] = indice_conf(i,j)


# In[21]:


get_ipython().run_line_magic('matplotlib', 'qt5')
plt.imshow(confusion_matrix)
plt.colorbar()
plt.title('Confusion Matrix (2 ways) \n Ratio of misclassified samples from class A using NCM btw. Proto A and Proto B \n with every sample without EME preprocessing \n 600 shots = 600 queries')
plt.ylabel('Samples and Prototype from class (class A)')
plt.xlabel('Prototype from class (class B)')


# In[22]:


def indice_conf_raw(cl1,cl2):
    featcl1 = feat[cl1]
    featcl2 = feat[cl2]
    mean_cl1 = featcl1.mean(0)
    mean_cl2 = featcl2.mean(0)
    d11 = torch.cdist(featcl1,mean_cl1.unsqueeze(0))
    d12 = torch.cdist(featcl1,mean_cl2.unsqueeze(0))
    d21 = torch.cdist(featcl2,mean_cl1.unsqueeze(0))
    d22 = torch.cdist(featcl2,mean_cl2.unsqueeze(0))
    bad1 = d12<d11
    bad2 = d21<d22
    return bad1.sum()/600,bad2.sum()/600   # combien de la classe 1 sont mal classé sachant qu'ils sont confronté à la classe 2 et inversement


# In[23]:


confusion_matrix_raw = torch.zeros((100,100))
for i in tqdm(range(0,100)):
    for j in range(i,100):
        a,b =indice_conf_raw(i,j)
        confusion_matrix_raw[i,j] , confusion_matrix_raw[j,i] = a,b


# In[24]:


plt.imshow(confusion_matrix_raw)
plt.colorbar()
plt.title('Confusion Matrix (2 ways) \n Ratio of misclassified samples from class A using NCM btw. Proto A and Proto B \n with every sample without EME preprocessing \n 600 shots = 600 queries')
plt.ylabel('Samples and Prototype from class (class A)')
plt.xlabel('Prototype from class (class B)')


# In[29]:


plt.imshow(confusion_matrix - confusion_matrix_raw)
plt.colorbar()
plt.title('Difference (EME or not) Confusion Matrix (2 ways) NCM classification error\n   \n 600 shots = 600 queries')
plt.ylabel('class with samples')
plt.xlabel('class with the adversary prototype')


# In[26]:


indice_conf_raw(31,80)


# In[27]:


indice_conf_raw(80,31)


# In[28]:


confusion_matrix[80,81]==confusion_matrix[81,80]

