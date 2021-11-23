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
from sklearn.metrics.pairwise import cosine_distances


# # Visualizing the Disregarding classes
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


# In[6]:


def proj_class(i,test_features,letter='A'):
    if letter=='A':
        #one projection per 64 clesses on miniimagenet
        w=base_mean[i]    #select weights of the i-th class
    else:
        w=base_meanB[i] 
    proj = torch.matmul(test_features,w)/ torch.norm(w)**2    #get coef of projection and normalize
    try:
        projection_ortho = proj.unsqueeze(-1).repeat(1,640)
    except:
        projection_ortho = proj.unsqueeze(-1).repeat(1,1,640)
    projection_ortho = projection_ortho * w                   #vector of projection along w 
    projection_ortho = test_features - projection_ortho       #projection on the orthogonal space of w
    return projection_ortho


# In[7]:


feat_20 = proj_class(50,feat.mean(1),letter='A')
plt.imshow(torch.cdist(feat_20,feat.mean(1))-torch.cdist(feat.mean(1),feat.mean(1)))
plt.colorbar()


# In[8]:


feat_20 = proj_class(50,feat,letter='A')
plt.imshow(torch.cdist(feat_20.mean(1),feat.mean(1))-torch.cdist(feat.mean(1),feat.mean(1)))
plt.colorbar()


# In[9]:


filenametrain = '/home/r21lafar/Documents/dataset/miniimagenetimages/train.csv'
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
def openimg(cl,title):
    if cl<64:
        src=train
    if cl>=80:
        src=test
        cl-=80
    if type(cl)==int:
        plt.figure(figsize=(5,5))
        idx=int((cl+0.5)*600)+np.random.randint(-100,100)
        filename=src[idx][0]
        im = Image.open(directory +filename)
        plt.title(title)
        plt.imshow(np.array(im))


# In[10]:


def distance_from_base(proj,run,plot=False,letter='A'):
    if letter=='A':
        fs_run = feat[acc[0,0,run].long()]
    else:
        fs_run = featB[acc[0,0,run].long()]
    if proj==-1 and run ==-1:
        if letter=='A':
            proto_fs = feat[-20:].mean(1)
        else:
            proto_fs = featB[-20:].mean(1)
    else:
        fs_run = torch.gather(fs_run,dim=1,index=idx[0,run].unsqueeze(-1).repeat(1,1,640).long()) 
        proto_fs = fs_run[:,:shot].mean(1)
    if proj!=0:
        proto_fs=proj_class(proj-1,proto_fs,letter=letter)
    if letter=='A': 
        D = torch.cdist(proto_fs,base_mean)
    else:
        D = torch.cdist(proto_fs,base_meanB)
    if plot:
        plt.figure()
        plt.imshow(D.detach().numpy(),aspect='auto')
        plt.colorbar()
        plt.title('distance between FS class mean and base class '+letter+' mean \n (whole base dataset) projection ' +str(proj) + ' (0 is no projection)')
        plt.xlabel('64 base class mean')
        plt.ylabel('FS prototype of class')
    return D


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
da,db = distance_from_base(-1,-1,letter='A',plot=True),distance_from_base(-1,-1,letter='B',plot=True)
plt.figure()
plt.title('difference between backbone A and B on novel dataset means \n projection on base dataset means')
plt.imshow(da-db,aspect='auto')
plt.colorbar()


# ### Visualization of datasets with U-map

# In[12]:


a=umap.UMAP().fit_transform(feat.mean(1))
b=umap.UMAP().fit_transform(featB.mean(1))


# In[13]:


plt.plot(a[:64,0],a[:64,1],'.',label='base')
plt.plot(a[-20:,0],a[-20:,1],'.',label='novel')
plt.legend()
example = list(range(64)) + list(range(80,100))
label = [str(i) for i in example]
for i in range(len(label)):
    plt.annotate(label[i], (a[example[i],0], a[example[i],1]))


# In[14]:


plt.plot(a[:64,0],a[:64,1],'.',label='base')
plt.plot(a[-20:,0],a[-20:,1],'.',label='novel')
plt.legend()
example = [57,96,37,92,47,87,88,16]
label = [str(i) for i in example]
label2 = ['roadsign','scoreboard','clarinette','electric guitar','spider net', 'ant', 'furet', 'small dog' ]
for i in range(len(label)):
    plt.annotate(label2[i], (a[example[i],0], a[example[i],1]))


# In[15]:


plt.plot(b[:64,0],b[:64,1],'.',label='base')
plt.plot(b[-20:,0],b[-20:,1],'.',label='novel')
plt.legend()
example = [57,96,37,92,47,87,88,16]
label = [str(i) for i in example]
label2 = ['roadsign','scoreboard','clarinette','electric guitar','spider net', 'ant', 'furet', 'small dog' ]
for i in range(len(label)):
    plt.annotate(label2[i], (b[example[i],0], b[example[i],1]))


# In[16]:


openimg(16,'')


# ## Create FS scenarii or runs 
# ### 2 ways

# In[17]:


n_runs, batch_few_shot_runs = 200,10
n_ways=5
def ncm(train_features, features, run_classes, run_indices, n_shots,i_proj):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(n_ways).unsqueeze(1).unsqueeze(0)
        #features = preprocess(train_features, features)
        scores = []
        score=0
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            distances = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, n_ways, 1, dim), dim = 4, p = 2)
            winners = torch.min(distances, dim = 2)[1]
            accuracy = (winners == targets)
            if batch_idx==0:
                full_accuracy=accuracy
                full_mean=means
            else:
                full_accuracy=torch.cat((full_accuracy,accuracy),dim=0)
                full_mean=torch.cat((full_mean,means),dim=0)
        return full_accuracy,full_mean

    
def generate_runs(data, run_classes, run_indices, batch_idx):
    n_runs, n_ways, n_samples = run_classes.shape[0], run_classes.shape[1], run_indices.shape[2]
    run_classes = run_classes[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs]
    run_indices = run_indices[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs]
    run_classes = run_classes.unsqueeze(2).unsqueeze(3).repeat(1,1,data.shape[1], data.shape[2])
    run_indices = run_indices.unsqueeze(3).repeat(1, 1, 1, data.shape[2])
    datas = data.unsqueeze(0).repeat(batch_few_shot_runs, 1, 1, 1)
    cclasses = torch.gather(datas, 1, run_classes.to(torch.int64))
    res = torch.gather(cclasses, 2, run_indices)
    return res

def define_runs(n_ways, n_shots, n_queries, num_classes, elements_per_class):
    shuffle_classes = torch.LongTensor(np.arange(num_classes))
    run_classes = torch.LongTensor(n_runs, n_ways)
    run_indices = torch.LongTensor(n_runs, n_ways, n_shots + n_queries)
    for i in range(n_runs):
        run_classes[i] = torch.randperm(num_classes)[:n_ways]
        for j in range(n_ways):
            run_indices[i,j] = torch.randperm(elements_per_class[run_classes[i, j]])[:n_shots + n_queries]
    return run_classes, run_indices


# In[18]:


run_classes, run_indices = define_runs(n_ways, 5, 500,20, [600 for i in range(20)])


# In[19]:



A,_ = ncm(feat[:64], feat[-20:], run_classes, run_indices, 5,0)
B,_ = ncm(featB[:64], featB[-20:],run_classes, run_indices, 5,0)
plt.plot(A.float().mean(-1).mean(-1),label='backbone A')
plt.plot(B.float().mean(-1).mean(-1),label='backbone B')
plt.legend()
plt.xlabel('run')
plt.ylabel('accuracy')
plt.title('no projection')


# In[20]:


for i in tqdm(range(65)):
    if i!=0:
        feature=proj_class(i-1,feat,'A')
        featureB=proj_class(i-1,featB,'B')
    else:
        feature =feat
        featureB =featB
    A,meanA = ncm(feature[:64], feature[-20:], run_classes, run_indices, 5,0)
    B,meanB = ncm(featureB[:64], featureB[-20:],run_classes, run_indices, 5,0)
    if i==0:
        fullA = A.unsqueeze(0)
        fullB = B.unsqueeze(0)
        fullmeanA = meanA.unsqueeze(0)
        fullmeanB = meanB.unsqueeze(0)
    else:
        fullA = torch.cat((fullA, A.unsqueeze(0)) ,dim = 0)
        fullB = torch.cat((fullB, B.unsqueeze(0)) ,dim = 0)
        fullmeanA = torch.cat((fullmeanA, meanA.unsqueeze(0)) ,dim = 0)
        fullmeanB = torch.cat((fullmeanB, meanB.unsqueeze(0)) ,dim = 0)


# In[21]:


def what_proj(run):
    return fullA[:,run].float().mean(-1).mean(-1).argsort()-1


# In[22]:


baseline = fullA[0].float().mean(-1).mean(-1)
best_acc = fullA[1:].float().mean(-1).mean(-1).max(dim = 0)
best_boost = best_acc[0] - baseline


# In[23]:


plt.plot(baseline,best_boost,'.')
plt.xlabel('baseline')
plt.ylabel('best boost' ) 


# In[24]:


plt.hist(best_boost.detach().numpy(),bins=20)
plt.xlabel('best boost')
plt.ylabel('frequency')
plt.title('64 base vectors 500 runs')


# In[32]:


run = 0
featb1 = generate_runs(feat, run_classes, run_indices, 0)
feature = featb1[run,:5,:5].reshape(-1,640)
plt.figure()
plt.imshow(cosine_distances(feature, base_mean))
plt.colorbar()
plt.figure()

plt.figure()
plt.plot(fullA[:,run].float().mean(-1).mean(-1),'.')
plt.hlines(y=fullA[0,run].float().mean(),xmin = 0 ,xmax =64
           ,label='baseline no proj')
plt.xlabel('projection (0 is no projection)')
plt.ylabel('accuracy')
plt.legend()


# In[27]:


run=11
acc_run = fullA[1:,run].float().mean(-1).mean(-1)
var_cs = cosine_distances(feature, base_mean).var(0)
plt.plot(var_cs,acc_run,'.')
plt.hlines(y=fullA[0,run].float().mean(),xmin = var_cs.min() ,xmax =var_cs.max()
           ,label='baseline no proj')
plt.legend()


# In[38]:


get_ipython().run_line_magic('matplotlib', 'qt5')
run = 12
nb_sample=30
mk_size=4
plt.figure()
plt.plot(fullA[:,run].float().mean(-1).mean(-1))

FULLumap = torch.cat((base_mean,fullmeanA[0,run],feat[80+run_classes[run],:nb_sample].reshape(n_ways*nb_sample,640) ))
umapA=umap.UMAP().fit_transform(FULLumap)
plt.figure()
plt.plot(umapA[:64,0],umapA[:64,1],'o',label='base', c='b')
plt.plot(umapA[64,0],umapA[64,1],'*',label='proto 0', c='purple',markersize=20)
plt.plot(umapA[65,0],umapA[65,1],'*',label='proto 1', c='k',markersize=20)

plt.plot(umapA[69:69+nb_sample,0],umapA[64+5:69+nb_sample,1],'.',label='samples 0',markersize=mk_size, c='purple')
plt.plot(umapA[64+5+nb_sample:69+nb_sample*2,0],umapA[64+5+nb_sample:69+nb_sample*2,1],'.',label='samples 1',markersize=mk_size, c='k')

plt.legend()

boost = fullA[:,run].float().mean(-1).mean(-1)-fullA[0,run].float().mean(-1).mean(-1)
example = what_proj(run)
signboost = boost>=0.
label = [str(i) for i in range(65)]
couleur = ['red','green']
for i in range(len(label)):
    plt.annotate(label[i], (umapA[example[i],0], umapA[example[i],1]), color = couleur[signboost[example[i]]*1])


# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')
run = 0
plt.plot(fullA[:,run].float().mean(-1).mean(-1),label='backbone A')
plt.plot(fullB[:,run].float().mean(-1).mean(-1),label='backbone B')
plt.legend()
plt.xlabel('projection')
plt.ylabel('accuracy')
print(fullA[:,run].shape)


# In[37]:


best, worst = fullA[:,run].float().mean(-1).mean(-1).argmax().item()-1 ,fullA[:,run].float().mean(-1).mean(-1).argmin().item()-1
print('best',best)
print('worst',worst)
listsort = fullA[:,run].float().mean(-1).mean(-1).argsort()-1
print(listsort)


# In[22]:


FULLumap = torch.cat((base_mean,fullmeanA[0,run]))
umapA=umap.UMAP().fit_transform(FULLumap)
plt.plot(umapA[:-5,0],umapA[:-5,1],'.',label='base')
plt.plot(umapA[-5:,0],umapA[-5:,1],'.',label='no proj')
plt.legend()
example = [best,worst]
label = ['best', 'worst']
for i in range(len(label)):
    plt.annotate(label[i], (umapA[example[i],0], umapA[example[i],1]))


# In[23]:


FULLumap = torch.cat((proj_class(best,base_mean,'A'),fullmeanA[best+1,run]))
umapA=umap.UMAP().fit_transform(FULLumap)
plt.plot(umapA[:-5,0],umapA[:-5,1],'.',label='base')
plt.plot(umapA[-5:,0],umapA[-5:,1],'.',label='best proj')
plt.legend()
example = [best,worst]
label = ['best', 'worst']
for i in range(len(label)):
    plt.annotate(label[i], (umapA[example[i],0], umapA[example[i],1]))


# In[24]:


FULLumap = torch.cat((proj_class(worst,base_mean,'A'),fullmeanA[worst+1,run]))
umapA=umap.UMAP().fit_transform(FULLumap)
plt.plot(umapA[:-5,0],umapA[:-5,1],'.',label='base')
plt.plot(umapA[-5:,0],umapA[-5:,1],'.',label='worst proj')
plt.legend()
example = [best,worst]
label = ['best', 'worst']
for i in range(len(label)):
    plt.annotate(label[i], (umapA[example[i],0], umapA[example[i],1]))


# In[372]:


feat[run_classes[run],:30].shape


# In[26]:


get_ipython().run_line_magic('matplotlib', 'qt5')
nb_sample=30
mk_size=3
FULLumap = torch.cat((base_mean,fullmeanA[0,run],feat[80+run_classes[run],:nb_sample].reshape(n_ways*nb_sample,640) ))
umapA=umap.UMAP().fit_transform(FULLumap)
plt.plot(umapA[:64,0],umapA[:64,1],'.',label='base', c='b')
plt.plot(umapA[64,0],umapA[64,1],'.',label='proto 0', c='orange')
plt.plot(umapA[65,0],umapA[65,1],'.',label='proto 1', c='g')
plt.plot(umapA[66,0],umapA[66,1],'.',label='proto 2', c='y')
plt.plot(umapA[67,0],umapA[67,1],'.',label='proto 3', c='k')
plt.plot(umapA[68,0],umapA[68,1],'.',label='proto 4', c='purple')
plt.plot(umapA[69:69+nb_sample,0],umapA[64+5:69+nb_sample,1],'.',label='samples 0',markersize=mk_size, c='orange')
plt.plot(umapA[64+5+nb_sample:69+nb_sample*2,0],umapA[64+5+nb_sample:69+nb_sample*2,1],'.',label='samples 1',markersize=mk_size, c='g')
plt.plot(umapA[64+5+nb_sample*2:69+nb_sample*3,0],umapA[64+5+nb_sample*2:69+nb_sample*3,1],'.',label='samples 2',markersize=mk_size, c='y')
plt.plot(umapA[64+5+nb_sample*3:69+nb_sample*4,0],umapA[64+5+nb_sample*3:69+nb_sample*4,1],'.',label='samples 3',markersize=mk_size, c='k')
plt.plot(umapA[64+5+nb_sample*4:69+nb_sample*5,0],umapA[64+5+nb_sample*4:69+nb_sample*5,1],'.',label='samples 4',markersize=mk_size, c='purple')
plt.legend()
example = listsort
label = ['best', 'worst']
label = [ str(i) for i in  range(listsort.shape[0])]
plt.title('no projection')
for i in range(len(label)):
    plt.annotate(label[i], (umapA[example[i],0], umapA[example[i],1]))


# In[35]:


def what_proj(boost,run):
    return boost.argsort()


# In[502]:


get_ipython().run_line_magic('matplotlib', 'inline')
run=12
plt.figure()
plt.plot(fullA[:,run].float().mean(-1).mean(-1))


# In[36]:


get_ipython().run_line_magic('matplotlib', 'qt5')
run = 12
nb_sample=30
mk_size=4
plt.figure()
plt.plot(fullA[:,run].float().mean(-1).mean(-1))

FULLumap = torch.cat((base_mean,fullmeanA[0,run],feat[80+run_classes[run],:nb_sample].reshape(n_ways*nb_sample,640) ))
umapA=umap.UMAP().fit_transform(FULLumap)
plt.figure()
plt.plot(umapA[:64,0],umapA[:64,1],'o',label='base', c='b')
plt.plot(umapA[64,0],umapA[64,1],'*',label='proto 0', c='purple',markersize=20)
plt.plot(umapA[65,0],umapA[65,1],'*',label='proto 1', c='k',markersize=20)

plt.plot(umapA[69:69+nb_sample,0],umapA[64+5:69+nb_sample,1],'.',label='samples 0',markersize=mk_size, c='purple')
plt.plot(umapA[64+5+nb_sample:69+nb_sample*2,0],umapA[64+5+nb_sample:69+nb_sample*2,1],'.',label='samples 1',markersize=mk_size, c='k')

plt.legend()

boost = fullA[:,run].float().mean(-1).mean(-1)-fullA[0,run].float().mean(-1).mean(-1)
example = what_proj(boost,run)
signboost = boost>=0.
label = [str(i) for i in range(65)]
couleur = ['red','green']
for i in range(len(label)):
    plt.annotate(label[i], (umapA[example[i],0], umapA[example[i],1]), color = couleur[signboost[example[i]]*1])


# In[515]:


get_ipython().run_line_magic('matplotlib', 'qt5')
run = 6
nb_sample=30
mk_size=4
plt.figure()
plt.plot(fullA[:,run].float().mean(-1).mean(-1))

FULLumap = torch.cat((base_mean,fullmeanA[0,run],feat[80+run_classes[run],:nb_sample].reshape(n_ways*nb_sample,640) ))
umapA=umap.UMAP().fit_transform(FULLumap)
plt.figure()
plt.plot(umapA[:64,0],umapA[:64,1],'o',label='base', c='b')
plt.plot(umapA[64,0],umapA[64,1],'*',label='proto 0', c='purple',markersize=20)
plt.plot(umapA[65,0],umapA[65,1],'*',label='proto 1', c='k',markersize=20)

plt.plot(umapA[69:69+nb_sample,0],umapA[64+5:69+nb_sample,1],'.',label='samples 0',markersize=mk_size, c='purple')
plt.plot(umapA[64+5+nb_sample:69+nb_sample*2,0],umapA[64+5+nb_sample:69+nb_sample*2,1],'.',label='samples 1',markersize=mk_size, c='k')

plt.legend()

boost = fullA[:,run].float().mean(-1).mean(-1)-fullA[0,run].float().mean(-1).mean(-1)
example = what_proj(boost,run)
signboost = boost>=0.
label = [str(i) for i in range(65)]
couleur = ['red','green']
for i in range(len(label)):
    plt.annotate(str(np.round(100*boost[example[i]].detach().numpy(),3)), (umapA[example[i],0], umapA[example[i],1]), color = couleur[signboost[example[i]]*1])


# In[470]:


print(signboost)


# In[453]:


get_ipython().run_line_magic('matplotlib', 'qt5')
nb_sample=30
mk_size=3

FULLumap = torch.cat((proj_class(best,base_mean,'A'),fullmeanA[best+1,run],proj_class(best,feat[80+run_classes[run],:nb_sample].reshape(5*nb_sample,640),'A') ))
umapA=umap.UMAP().fit_transform(FULLumap)
plt.plot(umapA[:64,0],umapA[:64,1],'.',label='base', c='b')
plt.plot(umapA[64,0],umapA[64,1],'.',label='proto 0', c='orange')
plt.plot(umapA[65,0],umapA[65,1],'.',label='proto 1', c='g')
plt.plot(umapA[66,0],umapA[66,1],'.',label='proto 2', c='y')
plt.plot(umapA[67,0],umapA[67,1],'.',label='proto 3', c='k')
plt.plot(umapA[68,0],umapA[68,1],'.',label='proto 4', c='purple')
plt.plot(umapA[69:69+nb_sample,0],umapA[64+5:69+nb_sample,1],'.',label='samples 0',markersize=mk_size, c='orange')
plt.plot(umapA[64+5+nb_sample:69+nb_sample*2,0],umapA[64+5+nb_sample:69+nb_sample*2,1],'.',label='samples 1',markersize=mk_size, c='g')
plt.plot(umapA[64+5+nb_sample*2:69+nb_sample*3,0],umapA[64+5+nb_sample*2:69+nb_sample*3,1],'.',label='samples 2',markersize=mk_size, c='y')
plt.plot(umapA[64+5+nb_sample*3:69+nb_sample*4,0],umapA[64+5+nb_sample*3:69+nb_sample*4,1],'.',label='samples 3',markersize=mk_size, c='k')
plt.plot(umapA[64+5+nb_sample*4:69+nb_sample*5,0],umapA[64+5+nb_sample*4:69+nb_sample*5,1],'.',label='samples 4',markersize=mk_size, c='purple')
plt.legend()
example = listsort
label = ['best', 'worst']
plt.title('best')
boost = fullA[:,run].float().mean(-1).mean(-1)-fullA[0,run].float().mean(-1).mean(-1)
signboost = boost>=0.
couleur = ['red','green']
for i in range(len(label)):
    plt.annotate(label[i], (umapA[example[i],0], umapA[example[i],1]), color = couleur[signboost[i]*1])


# In[405]:


get_ipython().run_line_magic('matplotlib', 'qt5')
nb_sample=30
mk_size=3
FULLumap = torch.cat((proj_class(worst,base_mean,'A'),fullmeanA[worst+1,run],proj_class(worst,feat[80+run_classes[run],:nb_sample].reshape(5*nb_sample,640),'A') ))
umapA=umap.UMAP().fit_transform(FULLumap)
plt.plot(umapA[:64,0],umapA[:64,1],'.',label='base', c='b')
plt.plot(umapA[64,0],umapA[64,1],'.',label='proto 0', c='orange')
plt.plot(umapA[65,0],umapA[65,1],'.',label='proto 1', c='g')
plt.plot(umapA[66,0],umapA[66,1],'.',label='proto 2', c='y')
plt.plot(umapA[67,0],umapA[67,1],'.',label='proto 3', c='k')
plt.plot(umapA[68,0],umapA[68,1],'.',label='proto 4', c='purple')
plt.plot(umapA[69:69+nb_sample,0],umapA[64+5:69+nb_sample,1],'.',label='samples 0',markersize=mk_size, c='orange')
plt.plot(umapA[64+5+nb_sample:69+nb_sample*2,0],umapA[64+5+nb_sample:69+nb_sample*2,1],'.',label='samples 1',markersize=mk_size, c='g')
plt.plot(umapA[64+5+nb_sample*2:69+nb_sample*3,0],umapA[64+5+nb_sample*2:69+nb_sample*3,1],'.',label='samples 2',markersize=mk_size, c='y')
plt.plot(umapA[64+5+nb_sample*3:69+nb_sample*4,0],umapA[64+5+nb_sample*3:69+nb_sample*4,1],'.',label='samples 3',markersize=mk_size, c='k')
plt.plot(umapA[64+5+nb_sample*4:69+nb_sample*5,0],umapA[64+5+nb_sample*4:69+nb_sample*5,1],'.',label='samples 4',markersize=mk_size, c='purple')
plt.legend()
plt.title('worst')
example = listsort
label = ['best', 'worst']
label = [str(i) for i in range(listsort.shape[0])]


for i in range(len(label)):
    plt.annotate(label[i], (umapA[example[i],0], umapA[example[i],1]), )


# In[229]:


plt.plot(umapA[0,0],umapA[0,1],'.',label='no proj 0')
plt.plot(umapA[1,0],umapA[1,1],'.',label='best proj 0')
plt.plot(umapA[2,0],umapA[2,1],'.',label='worst proj 0')
plt.plot(umapA[3,0],umapA[3,1],'.',label='no proj 1')
plt.plot(umapA[4,0],umapA[4,1],'.',label='best proj 1')
plt.plot(umapA[5,0],umapA[5,1],'.',label='worst proj 1')
plt.plot(umapA[6,0],umapA[6,1],'.',label='no proj 2')
plt.plot(umapA[7,0],umapA[7,1],'.',label='best proj 2')
plt.plot(umapA[8,0],umapA[8,1],'.',label='worst proj 2')
plt.plot(umapA[9,0],umapA[9,1],'.',label='no proj 3')
plt.plot(umapA[10,0],umapA[10,1],'.',label='best proj 3')
plt.plot(umapA[11,0],umapA[11,1],'.',label='worst proj 3')
plt.plot(umapA[12,0],umapA[12,1],'.',label='no proj 4')
plt.plot(umapA[13,0],umapA[13,1],'.',label='best proj 4')
plt.plot(umapA[14,0],umapA[14,1],'.',label='worst proj 4')
plt.plot(umapA[15:,0],umapA[15:,1],'.',label='base')
plt.legend()


# In[10]:


d=distance_from_base(30,2000,plot=True)


# In[13]:


acc[0,1].mean()


# In[21]:


proj=55
run=1000
diff=distance_from_base(proj+1,run)-distance_from_base(0,run)
print('diff mean', diff.mean().item())
boost=acc[:,1,run].mean(-1)[proj+1]-acc[:,1,run].mean(-1)[0]
print('boost proj ',str(proj) ,': ', boost.item() )
plt.figure()
plt.imshow(diff.detach().numpy(),aspect='auto')
plt.colorbar()
plt.title('run '+str(run)+' diff proj '+str(proj)+' and no proj')
plt.xlabel('64 base class mean')
plt.ylabel('FS prototype of class')
plt.figure()
plt.plot(acc[:,1,run].mean(-1),'*')
plt.hlines(y=acc[:,1,run].mean(-1)[0],xmin=0,xmax=65)
plt.grid()


# In[196]:


plt.figure()
plt.imshow(diff23.detach().numpy()-diff25.detach().numpy(),aspect='auto')
plt.colorbar()


# In[109]:


proj=5
run=50
diff=distance_from_base(0,run,letter='B')-distance_from_base(proj,run,letter='B')
boost=acc[:,1,run].mean(-1)[0]-acc[:,1,run].mean(-1)[proj+1]
print('boost proj ',str(proj) ,': ', boost )
plt.figure()
plt.imshow(diff.detach().numpy(),aspect='auto')
plt.colorbar()
plt.title('run '+str(run)+' diff proj '+str(proj)+' and no proj')
plt.xlabel('64 base class mean')
plt.ylabel('FS prototype of class')
plt.figure()
plt.plot(accB[:,1,run].mean(-1),'*')
plt.hlines(y=accB[:,1,run].mean(-1)[0],xmin=0,xmax=65)
plt.grid()


# In[131]:


da,db = distance_from_base(-1,-1,letter='A',plot=True),distance_from_base(-1,-1,letter='B',plot=True)
plt.figure()
plt.title('difference between backbone A and B on novel dataset means \n projection on base dataset means')
plt.imshow(da-db,aspect='auto')
plt.colorbar()


# In[27]:


D64 = torch.cdist(base_mean,base_mean)
plt.figure()
plt.imshow(D64,aspect='auto')
plt.colorbar()


# In[30]:


Dclassifier = torch.cdist(classifier,classifier)
plt.figure()
plt.imshow(Dclassifier.detach().numpy(),aspect='auto')
plt.colorbar()


# In[16]:


def look_run(run,proj,acc=acc,feat=feat,sample=0):
    print(acc[proj,0,run])
    best_class = acc[:,1,run].mean(-1).argsort(descending=True)[:2]
    print(best_class)
    plt.figure()
    plt.plot(acc[0,1,run],'.',label='no projection')
    plt.plot(acc[best_class[0],1,run],'.',label='best '+str(best_class[0].int().item()-1))
    plt.plot(acc[best_class[1],1,run],'.',label='2nd best '+ str(best_class[1].int().item()-1))
    plt.xlabel('classe')
    plt.ylabel('accuracy')
    plt.legend()
    plt.figure()
    plt.plot(acc[:,1,run].mean(-1),'-.')
    plt.hlines(y=acc[0,1,run].mean(-1),xmin=0, xmax = 65)
    plt.xlabel('projection removed')
    plt.ylabel('accuracy')
    return acc[:,1,run].mean(-1)


# In[17]:


feat20 = look_run(13,0)


# In[6]:


class_mean = feat.mean(2)
print(class_mean.shape)


# In[7]:


D = torch.cdist(class_mean,class_mean).detach().numpy()


# In[8]:


plt.imshow(D[15]-D[0])
plt.colorbar()


# On a observé le changement de distance entre moyennes de classe pour différentes projections. Les projections ne peuvent que diminuer les ditances entre classes car une dimension a été retirée. La première projection rapproche la classe 15 de la classe 7.

# On essaie de trouver 2 runs avec les même classes pour voir si l'erreur est la même.

# In[67]:


def generate_FS(run,n_shot=5,n_ways=5, hyperplan=False,plot=False,plot15=None,tsne=None,projvar=False):
    targets = torch.arange(n_ways).unsqueeze(1).unsqueeze(0)
    classes = acc[0,0,run].int().long()
    feature = feat[:,classes]
    feature = feature[:,:,torch.randperm(600)]
    feature=feature[:,:,:20]
    if tsne!=None:
        plot_tsne(feature,0,n_ways)
        plot_tsne(feature,tsne,n_ways)
    dim = feature.shape[-1]
    means = torch.mean(feature[:,:,:n_shot], dim = 2)
    if hyperplan:
        VARDIST=hyperplan_fn(feature, means, plot)
        
    distances = torch.norm(feature[:,:,n_shot:].reshape(feature.shape[0], n_ways, 1, -1, dim) - means.reshape(feature.shape[0], 1, n_ways, 1, dim), dim = 4, p = 2)
    if plot15!=None:
        plt.figure()
        plt.xlabel('15 few shot query' )
        plt.ylabel('query from FS class 5 shot classes')
        plt.title('distance to mean of FS class ' + str(plot15) + ' for all examples')
        plt.imshow(distances[0,plot15,:,:].detach().numpy(),aspect='auto')
        plt.colorbar()
    winners = torch.min(distances, dim = 2)[1]
    accuracy = (winners == targets).float().mean(2)
    
    if plot:
        plt.figure()
        plt.plot(accuracy[0],'*')
        plt.figure()
        plt.xlabel('projection base class')
        plt.ylabel('accuracy')
        plt.plot(np.arange(65),accuracy.mean(-1),'-.')
        #plt.grid(which='both')
        plt.hlines(y=accuracy.mean(-1)[0],xmin=0, xmax = 65)
        plt.xticks(np.arange(0,65,3))
        if hyperplan:
            plt.figure()
            plt.plot(VARDIST.detach().numpy(),accuracy.mean(-1)[1:],'.')
            plt.hlines(y=accuracy.mean(-1)[0],xmin=0, xmax = 0.02)
            plt.xlabel('variance over 5 ways of distance of the 5 way wrt the projected base class')
            plt.ylabel('accuracy of the prediction with such projection')
    if projvar:
        print('hello')
        var,men=get_variance_over64(feature)
        plt.figure()
        plt.plot(var[0,classes].sum(0),accuracy.mean(-1)[1:],'.')
        plt.xlabel('sum var projection on base class')
        plt.ylabel('accuracy')
        plt.hlines(y=accuracy.mean(-1)[0],xmin=0, xmax = 1)

    return accuracy


# In[50]:


FS0 = generate_FS(100,plot=True,hyperplan=True,tsne=0)


# In[25]:


def get_variance_over64(feature_processed,n_shot=5,n_ways=5):
    projection = torch.matmul(feature_processed, classifier.T)
    var=torch.var(projection,dim=1).detach().numpy()
    men=torch.mean(projection,dim=1).detach().numpy()
    
    #plt.plot(var.sum(0))
    return var,men


# In[26]:


som=np.zeros((64), 'float32')
somm=np.zeros((64), 'float32')
for i in tqdm(range(100)):
    classes = acc[0,0,0].int().long()
    feature = feat[:,classes]
    feature = feature[:,:,torch.randperm(600)]
    feature=feature[0,:,:5]
    var,men=get_variance_over64(feature)
    som+=var.sum(0)
    somm+=men.sum(0)
plt.figure()
plt.title('variance')
plt.plot(som)
plt.figure()
plt.title('mean')
plt.plot(somm)


# In[27]:


som=np.zeros((64), 'float32')
somm=np.zeros((64), 'float32')
for i in tqdm(range(100)):
    classes = acc[0,0,0].int().long()
    feature = feat[:,classes]
    feature = feature[:,:,torch.randperm(600)]
    feature=feature[0,:,:5]
    var,men=get_variance_over64(feature)
    som+=var.sum(0)
    somm+=men.sum(0)
plt.figure()
plt.title('variance of novel class on base classe (100 times)')
plt.plot(som)
plt.figure()
plt.title('mean of novel class on base classe (100 times)')
plt.plot(somm)


# In[28]:


som=np.zeros((64), 'float32')
somm=np.zeros((64), 'float32')
for i in tqdm(range(100)):
    classes = acc[0,0,0].int().long()
    feature = feat[:,classes]
    feature = feature[:,:,torch.randperm(600)]
    feature=feature[0,:,:5]
    var,men=get_variance_over64(feature)
    som+=var.sum(0)
    somm+=men.sum(0)
plt.figure()
plt.title('variance')
plt.plot(som)
plt.figure()
plt.title('mean')
plt.plot(somm)


# In[33]:


som=np.zeros((64), 'float32')
somm=np.zeros((64), 'float32')
for i in tqdm(range(1)):
    classes = acc[0,0,0].int().long()
    feature = feat[:,classes]
    feature = feature[:,:,torch.randperm(600)]
    feature=feature[0,:,:5]
    var,men=get_variance_over64(feature)
    som+=var.sum(0)
    somm+=men.sum(0)
plt.figure()
plt.title('variance')
plt.plot(som)
plt.figure()
plt.title('mean')
plt.plot(somm)


# A single 5 shot FS does not give the same class where variance is maximized as when tested on many samples (100 FS trials). Therefore the information would not be adapted for the 15 test samples of the FS trial ?! 

# 
# 
# Let's now try with some the hyperplan finding. 

# In[52]:


FS0 = generate_FS(100,plot=True,hyperplan=True)


# In[68]:


FS0 = generate_FS(100,projvar=True)


# In[41]:


filenametrain = '/home/r21lafar/Documents/dataset/miniimagenetimages/train.csv'
filenametest = '/home/r21lafar/Documents/dataset/miniimagenetimages/test.csv'
directory = '/home/r21lafar/Documents/dataset/miniimagenetimages/images/'

def extract_best_proj(acc,run,number_of_best=1):
    if number_of_best==1:
        return acc[:,run,:].mean(1).max().item(),acc[:,run,:].mean(1).argmax().item()-1
    else:
        sort=acc[:,run,:].mean(1).sort(descending=True)
        return sort[0][:number_of_best],sort[1].int()[:number_of_best]-1

def openimage(source, cl,title):
    if source=='test':
        src=test
    else:
        src=train
    if type(cl)==int:
        plt.figure(figsize=(5,5))
        idx=int((cl+0.5)*600)+np.random.randint(-100,100)
        filename=src[idx][0]
        im = Image.open(directory +filename)
        plt.title(title)
        plt.imshow(np.array(im))
    else:
        fig = plt.figure(figsize=(8,8))
        spec = mpl.gridspec.GridSpec(ncols=3, nrows=2)
        ax1 = fig.add_subplot(spec[0,0])   #classe 0
        ax2 = fig.add_subplot(spec[0,1])  #classe 1
        ax3 = fig.add_subplot(spec[0,2])  # classe 2
        ax4 = fig.add_subplot(spec[1,0])
        ax5 = fig.add_subplot(spec[1,1])
        L=[ax1,ax2,ax3,ax4,ax5]
        for i,classe in enumerate(cl):
            idx=int((classe+0.5)*600)+np.random.randint(-100,100)
            filename=src[idx][0]
            im = Image.open(directory +filename)
            L[i].imshow(np.array(im))
            L[i].set_title(str(i+1)+';'+str(classe))
            
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

def get_run(classes,acc,run,number_of_best=1):
    class_run=classes[0,run].int().tolist()
    baseline=acc[0,run,:].mean()
    best,best_class=extract_best_proj(acc,run,number_of_best)
    print(best)
    boostbest=best-baseline.repeat(number_of_best)
    print('best_class',best_class)
    print('boostbest',boostbest)
    print(class_run)
    return best_class,boostbest,class_run
            
def run_images_interpolation(classes,acc,run,test,train,choice=0):
    best_class,boostbest,class_run = get_run(classes,acc,run,10)
    best_class=best_class[choice].item()
    boostbest=boostbest[choice].item()
    openimage(test,train,'train', best_class, 'best class to remove')
    openimage(test,train,'test', class_run, 'best class to remove')


# In[59]:


def openimg(cl,title):
    if cl<64:
        src=train
    if cl>=80:
        src=test
        cl-=80
    if type(cl)==int:
        plt.figure(figsize=(5,5))
        idx=int((cl+0.5)*600)+np.random.randint(-100,100)
        filename=src[idx][0]
        im = Image.open(directory +filename)
        plt.title(title)
        plt.imshow(np.array(im))

