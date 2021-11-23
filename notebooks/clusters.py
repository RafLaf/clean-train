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


def sphering(features):
    return features / torch.norm(features, p = 2, dim = 2, keepdim = True)

def centering(train_features, features):
    return features - train_features.reshape(-1, train_features.shape[2]).mean(dim = 0).unsqueeze(0).unsqueeze(0)
feat_processed = sphering(centering(sphering(feat)[:64],sphering(feat) )) 


# In[7]:


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


# In[8]:


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


# In[9]:


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


# ## Create FS scenarii or runs 
# ### 2 ways

# In[10]:


n_runs, batch_few_shot_runs = 20,10
n_ways=2
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
            var_intra = runs[:,:,:n_shots].var(2).mean(-1)
            var_inter = runs[:,:,:n_shots].mean(2).var(1).mean(-1).unsqueeze(1)
            var = torch.cat((var_intra,var_inter),dim=1)
            distances = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, n_ways, 1, dim), dim = 4, p = 2)
            winners = torch.min(distances, dim = 2)[1]
            accuracy = (winners == targets)
            if batch_idx==0:
                full_accuracy=accuracy
                full_mean=means
                full_var = var
            else:
                full_accuracy=torch.cat((full_accuracy,accuracy),dim=0)
                full_mean=torch.cat((full_mean,means),dim=0)
                full_var=torch.cat((full_var,var),dim=0)
        return full_accuracy,full_mean,full_var

    
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


run_classes, run_indices = define_runs(n_ways, 5, 500,20, [600 for i in range(20)])


# In[11]:


A,_,full_var = ncm(feat[:64], feat[-20:], run_classes, run_indices, 5,0)
B,_,full_var = ncm(featB[:64], featB[-20:],run_classes, run_indices, 5,0)
plt.plot(A.float().mean(-1).mean(-1),label='backbone A')
plt.plot(B.float().mean(-1).mean(-1),label='backbone B')
plt.legend()
plt.xlabel('run')
plt.ylabel('accuracy')
plt.title('no projection')


# In[12]:


for i in tqdm(range(65)):
    if i!=0:
        feature=proj_class(i-1,feat,'A')
        featureB=proj_class(i-1,featB,'B')
    else:
        feature =feat
        featureB =featB
    A,meanA,varA = ncm(feature[:64], feature[-20:], run_classes, run_indices, 5,0)
    B,meanB,varB = ncm(featureB[:64], featureB[-20:],run_classes, run_indices, 5,0)
    if i==0:
        fullA = A.unsqueeze(0)
        fullB = B.unsqueeze(0)
        fullmeanA = meanA.unsqueeze(0)
        fullmeanB = meanB.unsqueeze(0)
        fullvarA = varA.unsqueeze(0)
        fullvarB = varB.unsqueeze(0)
    else:
        fullA = torch.cat((fullA, A.unsqueeze(0)) ,dim = 0)
        fullB = torch.cat((fullB, B.unsqueeze(0)) ,dim = 0)
        fullmeanA = torch.cat((fullmeanA, meanA.unsqueeze(0)) ,dim = 0)
        fullmeanB = torch.cat((fullmeanB, meanB.unsqueeze(0)) ,dim = 0)
        fullvarA = torch.cat((fullvarA, varA.unsqueeze(0)) ,dim = 0)
        fullvarB = torch.cat((fullvarB, varB.unsqueeze(0)) ,dim = 0)


# In[12]:


def what_proj(run):
    return fullA[:,run].float().mean(-1).mean(-1).argsort()-1


# In[13]:


fullA[0,2,0].float().mean(-1)


# In[14]:


run=0
fullvarA[0,run,:2].mean(-1)-fullvarA[0,run,2]


# In[15]:


for prj in [0,1,2,3]:
    plt.plot(fullvarA[prj,:,:2].mean(-1)-fullvarA[prj,:,2],fullA[prj,:,:].float().mean(-1).mean(-1),'.',label='projection '+ str(prj))
plt.xlabel('intraclass var -(minus)- interclass var')
plt.ylabel('accuracy of run')
plt.legend()
plt.title('20 runs')


# In[16]:


best_boost =fullA.float().mean(-1).mean(-1).max(0)[0] - fullA[0,:,:].float().mean(-1).mean(-1)
worst_boost =fullA.float().mean(-1).mean(-1).min(0)[0] - fullA[0,:,:].float().mean(-1).mean(-1)


# In[17]:


best_boost_id = fullA[:,:,:].float().mean(-1).mean(-1).max(0)[1]
worst_boost_id = fullA[:,:,:].float().mean(-1).mean(-1).min(0)[1]


# In[18]:


intrater = fullvarA[:,:,:2].mean(-1)-fullvarA[:,:,2]
intrater_min = intrater.min(0)[1]
intrater_max = intrater.max(0)[1]


# In[19]:


boost = torch.zeros(intrater_min.shape)
for i in range(intrater_min.shape[0]):
    boost[i] = fullA[intrater_min[i],i].float().mean(-1).mean(-1)-fullA[0,i].float().mean(-1).mean(-1)


# In[20]:


boost_max = torch.zeros(intrater_min.shape)
for i in range(intrater_min.shape[0]):
    boost_max[i] = fullA[intrater_max[i],i].float().mean(-1).mean(-1)-fullA[0,i].float().mean(-1).mean(-1)


# In[21]:


fullA.shape


# In[ ]:





# In[22]:



plt.hlines(y=0 ,xmin=0,xmax = 20)
plt.plot(boost,'.',label='proj with min intra - inter')
plt.plot(boost_max,'.',label='proj with max intra - inter')
plt.plot(best_boost,'.',label='best boost')
plt.xlabel('run')
plt.ylabel('boost')
plt.legend()


# In[23]:


intrater_best_boost = torch.zeros(intrater_min.shape)
for i in range(intrater_min.shape[0]):
    intrater_best_boost[i] = intrater[best_boost_id[i],i]


# In[24]:


plt.plot(intrater_best_boost,'.', label = 'best boost')
plt.plot(intrater.mean(0),'.', label = 'mean intra -inter')
plt.plot(intrater.min(0)[0],'.', label = 'minimum intra -inter')
plt.plot(intrater.max(0)[0],'.', label = 'maximum intra -inter')
plt.ylabel('intra-class - interclass variance')
plt.xlabel('run')
plt.legend()


# In[25]:


intrater.min(dim=0)[0]


# In[ ]:






















# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')
run = 12
nb_sample=30
mk_size=4
plt.figure()
plt.plot(fullA[:,run].float().mean(-1).mean(-1))


plt.figure()
plt.plot(fullvarA[:,run].float().mean(-1).mean(-1))

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


# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
run = 0
plt.plot(fullA[:,run].float().mean(-1).mean(-1),label='backbone A')
plt.plot(fullB[:,run].float().mean(-1).mean(-1),label='backbone B')
plt.legend()
plt.xlabel('projection')
plt.ylabel('accuracy')
print(fullA[:,run].shape)


# In[28]:


feat.shape


# In[29]:


nb_samples = 100
feat_sil = feat[:,:nb_samples].reshape(-1,640)


# In[30]:


labels = torch.arange(0,100).unsqueeze(1).repeat(1,nb_samples).reshape(-1)


# In[31]:


sil = silhouette_samples(feat_sil,labels)


# In[32]:


sil_r = sil.reshape(100,nb_samples)


# In[33]:


plt.plot(sil_r.mean(1),'.')
plt.xlabel('class')
plt.ylabel('silhouette')
plt.vlines(x=64,ymin=sil_r.mean(1).min(),ymax = sil_r.mean(1).max())
plt.vlines(x=64+20,ymin=sil_r.mean(1).min(),ymax = sil_r.mean(1).max())


# In[34]:


feat.shape


# In[35]:


plt.plot(feat.var(1).mean(1),'.',label='intra class variance')
plt.hlines(y=feat.mean(1).var(0).mean(),xmin=0,xmax=100,label='interclass variance')
plt.legend()
plt.xlabel('class')
plt.ylabel('mean variance over features')
plt.title('whole dataset')


# ## Project on aligning vector

# In[11]:


def proj_vec(v_proj):
    proj = torch.matmul(features,v_proj)/ torch.norm(v_proj)**2    #get coef of projection and normalize
    return proj


# In[12]:


run_classes, run_indices = define_runs(n_ways, 5, 500,20, [600 for i in range(20)])
A,full_meanA,full_varA = ncm(feat[:64], feat[-20:], run_classes, run_indices, 5,0)
B,full_meanB,full_var = ncm(featB[:64], featB[-20:],run_classes, run_indices, 5,0)


# In[13]:


def ncm_proj(train_features, features, run_classes, run_indices, n_shots):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(n_ways).unsqueeze(1).unsqueeze(0)
        #features = preprocess(train_features, features)
        scores = []
        score=0
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            v_diff = (means[:,0]-means[:,1])
            #v_diff = torch.randn(means[:,0].shape)
            var_intra = runs[:,:,:n_shots].var(2).mean(-1)
            var_inter = runs[:,:,:n_shots].mean(2).var(1).mean(-1).unsqueeze(1)
            var = torch.cat((var_intra,var_inter),dim=1)
            proj_means =torch.zeros(means[:,:,0].shape)
            proj_runs =torch.zeros(runs[:,:,:,0].shape)

            for i in range(batch_few_shot_runs):
                proj_runs[i] = torch.matmul(v_diff[i], torch.swapaxes(runs[i],-1,-2))
                proj_means[i] = torch.matmul(v_diff[i], torch.swapaxes(means[i],-1,-2))
            distances = torch.norm(proj_runs[:,:,n_shots:].reshape(batch_few_shot_runs, n_ways, 1, -1, 1) - proj_means.reshape(batch_few_shot_runs, 1, n_ways, 1, 1), dim = 4, p = 2)
            winners = torch.min(distances, dim = 2)[1]
            accuracy = (winners == targets)
            if batch_idx==0:
                full_accuracy=accuracy
                full_mean=means
                full_var = var
            else:
                full_accuracy=torch.cat((full_accuracy,accuracy),dim=0)
                full_mean=torch.cat((full_mean,means),dim=0)
                full_var=torch.cat((full_var,var),dim=0)
        return full_accuracy,full_mean,full_var
    


# In[ ]:


n_runs = 1000
run_classes, run_indices = define_runs(n_ways, 5, 500,20, [600 for i in range(20)])

n_shots=5


# In[105]:




a,b,c = ncm_proj(feat[:64], feat[-20:], run_classes, run_indices, n_shots)
print(a.float().mean())


# In[104]:


a,b,c = ncm(feat[:64], feat[-20:], run_classes, run_indices, n_shots,0)
print(a.float().mean())


# Avec et sans projection sur l'axe reliant les deux protopypes ou templates. La performance reste la même

# ## test suppression de classe orthogonale à v_diff

# In[101]:


def ncm_del_otho_vdif(train_features, features, run_classes, run_indices, n_shots):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(n_ways).unsqueeze(1).unsqueeze(0)
        #features = preprocess(train_features, features)
        scores = []
        score=0
        for batch_idx in tqdm(range(n_runs // batch_few_shot_runs)):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            for i in range(3):
                runs,means = remove_the_class(runs)
            var_intra = runs[:,:,:n_shots].var(2).mean(-1)
            var_inter = runs[:,:,:n_shots].mean(2).var(1).mean(-1).unsqueeze(1)
            var = torch.cat((var_intra,var_inter),dim=1)
            distances = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, n_ways, 1, dim), dim = 4, p = 2)
            winners = torch.min(distances, dim = 2)[1]
            accuracy = (winners == targets)
            if batch_idx==0:
                full_accuracy=accuracy
                full_mean=means
                full_var = var
            else:
                full_accuracy=torch.cat((full_accuracy,accuracy),dim=0)
                full_mean=torch.cat((full_mean,means),dim=0)
                full_var=torch.cat((full_var,var),dim=0)
        return full_accuracy,full_mean,full_var
    
    
def remove_the_class(runs):
    means = torch.mean(runs[:,:,:n_shots], dim = 2)
    v_diff = (means[:,0]-means[:,1])  #axis between proto 0 and proto 1 
    proj_base = torch.zeros(batch_few_shot_runs,base_mean.shape[0])
    for j in range(batch_few_shot_runs):
        for i in range(base_mean.shape[0]):
            w = base_mean[i]
            proj_base[j,i]  = torch.torch.matmul(v_diff[j], w)/torch.norm(w)
    id_proj = abs(proj_base).min(1)[1]
    for j in range(batch_few_shot_runs):
        runs[j] = proj_class(id_proj[j],runs[j])
    means = torch.mean(runs[:,:,:n_shots], dim = 2)
    return runs,means


# In[ ]:





# In[99]:


a,b,c  = ncm_del_otho_vdif(feat[:64], feat[-20:], run_classes, run_indices, n_shots)
print(a.float().mean().item())


# In[100]:


a,b,c = ncm(feat_processed[:64], feat_processed[-20:], run_classes, run_indices, n_shots,0)
print(a.float().mean())
a,b,c = ncm_del_otho_vdif(feat_processed[:64], feat_processed[-20:], run_classes, run_indices, n_shots)
print(a.float().mean())


# ## Test du LDA / shrinkage

# In[15]:


def LDA(run,**kwargs):
    s = run.shape
    s_shots =  run[:,:n_shots].shape
    run_reshaped = run.reshape(s[0]*s[1],-1)
    run_reshaped_shots = run[:,:n_shots].reshape(s_shots[0]*s_shots[1],-1)
    target = torch.cat((torch.zeros(n_shots),torch.ones(n_shots)))

    clf = LinearDiscriminantAnalysis(**kwargs)
    clf.fit(run_reshaped_shots, target)
    out = torch.tensor(clf.transform(run_reshaped))
    out = out.reshape((s[0],s[1],-1))
    return out


# In[16]:


def ncm_lda(train_features, features, run_classes, run_indices, n_shots,dictlda):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(n_ways).unsqueeze(1).unsqueeze(0)
        #features = preprocess(train_features, features)
        scores = []
        score=0
        for batch_idx in tqdm(range(n_runs // batch_few_shot_runs)):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)

            runs_reduced  = torch.zeros((runs.shape[0],runs.shape[1],runs.shape[2],n_components))
            for i,run in enumerate(runs):
                runs_reduced[i] = LDA(run,**dictlda)
            means = torch.mean(runs_reduced[:,:,:n_shots], dim = 2)
            distances = torch.norm(runs_reduced[:,:,n_shots:].reshape(batch_few_shot_runs, n_ways, 1, -1, n_components) - means.reshape(batch_few_shot_runs, 1, n_ways, 1, n_components), dim = 4, p = 2)
            winners = torch.min(distances, dim = 2)[1]
            accuracy = (winners == targets)
            if batch_idx==0:
                full_accuracy=accuracy
                full_mean=means
            else:
                full_accuracy=torch.cat((full_accuracy,accuracy),dim=0)
                full_mean=torch.cat((full_mean,means),dim=0)
        return full_accuracy,full_mean,1


# In[102]:


n_components = 1
dictlda = {'solver' : 'eigen','shrinkage': 'auto' , 'n_components' : n_components}
a,b,c  = ncm_lda(feat[:64], feat[-20:], run_classes, run_indices, n_shots, dictlda)
print(a.float().mean().item())


# In[103]:


n_components = 1
dictlda = {'solver' : 'svd' , 'n_components' : n_components}
a,b,c  = ncm_lda(feat[:64], feat[-20:], run_classes, run_indices, n_shots, dictlda)
print(a.float().mean().item())


# # Let us do it again with EME preprocesing 

# In[ ]:



def sphering(features):
    return features / torch.norm(features, p = 2, dim = 2, keepdim = True)

def centering(train_features, features):
    return features - train_features.reshape(-1, train_features.shape[2]).mean(dim = 0).unsqueeze(0).unsqueeze(0)
feat_processed = sphering(centering(sphering(feat)[:64],sphering(feat) )) 


# In[106]:


a,b,c = ncm(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots,0)
print(a.float().mean().item())


# In[107]:


a,b,c = ncm_proj(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots)
print(a.float().mean().item())


# In[108]:


base_mean = feat[:64].mean(-2)
a,b,c  = ncm_del_otho_vdif(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots)  #without EME on base clases
print(a.float().mean().item())


# In[109]:


base_mean = feat_processed[:64].mean(-2)
a,b,c  = ncm_del_otho_vdif(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots)  #without EME on base clases
print(a.float().mean().item())


# In[110]:


n_components = 1
dictlda = {'solver' : 'eigen','shrinkage': 'auto' , 'n_components' : n_components}
a,b,c  = ncm_lda(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots, dictlda)
print(a.float().mean().item())


# In[111]:


n_components = 1
dictlda = {'solver' : 'svd' , 'n_components' : n_components}
a,b,c  = ncm_lda(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots, dictlda)
print(a.float().mean().item())


# Avec EME, on obtient un très legère amélioration (en projetant sur la classe la plus orthogonale à l'axe). Cette amélioration se trouve cenpendant dans la marge d'erreur.

# # more shots 

# In[ ]:


l_shots = range(1,100,5)
L=[]
for n_shots in tqdm(range(1,100,5)):
    a,b,c = ncm(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots,0)
    L.append(a.float().mean().item())


# In[ ]:


plt.plot(l_shots,L,'.')
plt.xlabel('n_shots')
plt.ylabel('mean accuracy')
plt.title('1000 runs 2-ways')


# In[26]:


n_shots = 5
a,b,c = ncm(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots,0)
print(a.float().mean().item())


# In[27]:


a,b,c = ncm_proj(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots)
print(a.float().mean().item())


# In[28]:


base_mean = feat_processed[:64].mean(-2)
a,b,c  = ncm_del_otho_vdif(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots)  #without EME on base clases
print(a.float().mean().item())


# In[29]:


n_components = 1
dictlda = {'solver' : 'eigen','shrinkage': 'auto' , 'n_components' : n_components}
a,b,c  = ncm_lda(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots, dictlda)
print(a.float().mean().item())


# In[30]:


n_components = 1
dictlda = {'solver' : 'svd' , 'n_components' : n_components}
a,b,c  = ncm_lda(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots, dictlda)
print(a.float().mean().item())


# ## Outlier management

# In[75]:


def ncm_pop(train_features, features, run_classes, run_indices, n_shots):
    global accuracy_s
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(n_ways).unsqueeze(1).unsqueeze(0)
        #features = preprocess(train_features, features)
        scores = []
        score=0
        for batch_idx in tqdm(range(n_runs // batch_few_shot_runs)):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            support = runs[:,:,:n_shots]
            means = torch.mean(support, dim = 2)

            distances_s = torch.norm(runs[:,:,:n_shots].reshape(batch_few_shot_runs, n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, n_ways, 1, dim), dim = 4, p = 2)
            
            winners_s = torch.min(distances_s, dim = 2)[1]
            accuracy_s = (winners_s == targets)
            for i in range(batch_few_shot_runs):
                for j in range(n_ways):
                    means[i,j]= torch.mean(support[i,j,accuracy_s[i,j]], dim = 0) 
                    
            #means = torch.mean(support, dim = 2)
            distances = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, n_ways, 1, dim), dim = 4, p = 2)
            winners = torch.min(distances, dim = 2)[1]
            accuracy = (winners == targets)
            if batch_idx==0:
                full_accuracy=accuracy
                full_accuracy_s=accuracy_s
                full_mean=means
            else:
                full_accuracy=torch.cat((full_accuracy,accuracy),dim=0)
                full_accuracy_s=torch.cat((full_accuracy_s,accuracy_s),dim=0)
                full_mean=torch.cat((full_mean,means),dim=0)

        
        return full_accuracy,full_mean,full_accuracy_s


# In[33]:


n_runs = 1000
run_classes, run_indices = define_runs(n_ways, 5, 500,20, [600 for i in range(20)])


# In[65]:


a,b,c = ncm(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots,0)
print(a.float().mean().item())


# In[76]:


a,b,c  = ncm_pop(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots)
print(a.float().mean().item(),c.float().mean().item())


# In[97]:


def ncm_2template(train_features, features, run_classes, run_indices, n_shots):
    global winners,winners2
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(n_ways).unsqueeze(1).unsqueeze(0)
        #features = preprocess(train_features, features)
        scores = []
        score=0
        for batch_idx in tqdm(range(n_runs // batch_few_shot_runs)):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            support = runs[:,:,:n_shots]
            means = torch.mean(support, dim = 2)
            means2 = torch.zeros(means.shape)

            distances_s = torch.norm(runs[:,:,:n_shots].reshape(batch_few_shot_runs, n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, n_ways, 1, dim), dim = 4, p = 2)
            
            winners_s = torch.min(distances_s, dim = 2)[1]
            accuracy_s = (winners_s == targets)
            for i in range(batch_few_shot_runs):
                for j in range(n_ways):
                    means[i,j]= torch.mean(support[i,j,accuracy_s[i,j]], dim = 0) 
                    if accuracy_s[0,0].sum().item() != 5 :
                        means2[i,j] = torch.mean(support[i,j,torch.logical_not(accuracy_s[i,j])], dim = 0) 
                    else:
                        means2[i,j] = means[i,j]
            
            distances = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, n_ways, 1, dim), dim = 4, p = 2)
            distances2 = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, n_ways, 1, -1, dim) - means2.reshape(batch_few_shot_runs, 1, n_ways, 1, dim), dim = 4, p = 2)

            winners = torch.min(distances, dim = 2)[1]
            winners2 = torch.min(distances2, dim = 2)[1]
            
            accuracy = torch.logical_or((winners == targets), (winners2 == targets))
            if batch_idx==0:
                full_accuracy=accuracy
                full_accuracy_s=accuracy_s
                full_mean=means
            else:
                full_accuracy=torch.cat((full_accuracy,accuracy),dim=0)
                full_accuracy_s=torch.cat((full_accuracy_s,accuracy_s),dim=0)
                full_mean=torch.cat((full_mean,means),dim=0)

        
        return full_accuracy,full_mean,full_accuracy_s


# In[83]:


a,b,c = ncm(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots,0)
print(a.float().mean().item())


# In[98]:


a,b,c  = ncm_2template(feat[:64], feat_processed[-20:], run_classes, run_indices, n_shots)
print(a.float().mean().item(),c.float().mean().item())

