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

# In[23]:


n_runs, batch_few_shot_runs = 20,10
n_ways=2
def ncm(train_features, features, run_classes, run_indices, n_shots):
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

n_shots = 5
run_classes, run_indices = define_runs(n_ways, n_shots, 600-n_shots,20, [600 for i in range(20)])


# In[11]:


A,_,full_var = ncm(feat[:64], feat[-20:], run_classes, run_indices, 5)
B,_,full_var = ncm(featB[:64], featB[-20:],run_classes, run_indices, 5)
plt.plot(A.float().mean(-1).mean(-1),label='backbone A')
plt.plot(B.float().mean(-1).mean(-1),label='backbone B')
plt.legend()
plt.xlabel('run')
plt.ylabel('accuracy')
plt.title('no projection')


# In[12]:


A.float().mean()


# In[24]:


for i in tqdm(range(65)):
    if i!=0:
        feature=proj_class(i-1,feat,'A')
        featureB=proj_class(i-1,featB,'B')
    else:
        feature =feat
        featureB =featB
    A,meanA,varA = ncm(feature[:64], feature[-20:], run_classes, run_indices, 5)
    B,meanB,varB = ncm(featureB[:64], featureB[-20:],run_classes, run_indices, 5)
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


# In[14]:


def what_proj(run):
    return fullA[:,run].float().mean(-1).mean(-1).argsort()-1


# In[15]:



modelA = torch.load('modelA1',map_location=torch.device('cpu'))


# In[16]:


biasA = modelA['linear.bias']
weightA = modelA['linear.weight']


# In[17]:


feat.shape


# In[18]:


logits = torch.einsum('cf,gsf->gsc',weightA,feat) + biasA


# In[19]:


A,_,full_var = ncm(logits[:64], logits[-20:], run_classes, run_indices, 5)


# In[20]:


A.float().mean()


# In[ ]:





# In[ ]:





# In[21]:


plt.plot(torch.mean(fullA.float(), dim =(1,2,3)),'.')
plt.hlines(y = torch.mean(fullA[0].float()), xmin = 0 , xmax = 65)
plt.title('projection performance'+str(n_shots)+' shots' + str(n_runs) + ' runs')
plt.xlabel('projection')
plt.ylabel('performance')


# In[ ]:





# In[ ]:





# In[ ]:





# # cosine distances n_shots =5

# In[62]:


logits = torch.einsum('cf,gsf->gsc',weightA,feat) + biasA
n_runs, batch_few_shot_runs = 20,10
n_ways=5
n_shots = 5
run_classes, run_indices = define_runs(n_ways, n_shots, 600-n_shots,20, [600 for i in range(20)])

for i in tqdm(range(65)):
    if i!=0:
        feature=logits.detach().clone()
        feature[:,:,i-1]=0
    else:
        feature =logits.detach().clone()
    A,meanA,varA = ncm(feature[:64], feature[-20:], run_classes, run_indices, n_shots)
    if i==0:
        fullA = A.unsqueeze(0)
        fullmeanA = meanA.unsqueeze(0)
        fullvarA = varA.unsqueeze(0)
    else:
        fullA = torch.cat((fullA, A.unsqueeze(0)) ,dim = 0)
        fullmeanA = torch.cat((fullmeanA, meanA.unsqueeze(0)) ,dim = 0)
        fullvarA = torch.cat((fullvarA, varA.unsqueeze(0)) ,dim = 0)
fullA5 = fullA.detach().clone()


# In[63]:


plt.plot(torch.mean(fullA.float(), dim =(1,2,3)),'.')
plt.hlines(y = torch.mean(fullA[0].float()), xmin = 0 , xmax = 65)
plt.title('projection performance'+str(n_shots)+' shots' + str(n_runs) + ' runs')
plt.xlabel('projection')
plt.ylabel('performance')


# In[51]:


print(run_indices.shape,run_classes.shape,logits.shape)
run0 = generate_runs(logits, run_classes, run_indices, 0)
base64 = torch.eye(base_mean.shape[0])


# ### cosine distance with base class + cosine distance with other shots

# In[64]:


run=3
get_ipython().run_line_magic('matplotlib', 'qt5')
cd = cosine_distances(run0[run,:,:n_shots].reshape(-1,64),base64)
plt.figure()
plt.imshow(cd)
plt.colorbar()


cd = cosine_distances(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))
plt.figure()
plt.imshow(cd)
plt.colorbar()

plt.figure()
plt.plot(torch.mean(fullA[1:,run].float(),dim=(1,2)),'.')
plt.hlines(y = torch.mean(fullA[0,run].float()), xmin = 0 , xmax = 65)
plt.grid()


# In[67]:


run0[run,:].shape


# In[118]:


plt.figure()
plt.imshow(run0[run,:,:5].reshape(-1,64))

def check_var(run_sample):
    intravar = run_sample[:,:n_shots].var(1)
    intervar = run_sample[:,:n_shots].mean(1).var(0)
    return intravar.sum(0)-intervar


# In[119]:


proj = 5

run_proj = run0[run].detach().clone()
run_proj[run,proj] = 0
check_var(run_proj)


# In[88]:


run_proj.shape


# In[120]:


L=[check_var(run0[run])]
for proj in range(64):
    run_proj = run0[run].detach().clone()
    run_proj[:,:,proj] = 0
    L.append(check_var(run_proj))


# In[171]:


print(run_indices.shape,run_classes.shape,logits.shape)
run0 = generate_runs(logits, run_classes, run_indices, 0)
base64 = torch.eye(base_mean.shape[0])


# In[172]:


L.mean(1).argsort()


# In[173]:


plt.figure()
plt.plot(torch.diag((L[1:]-L[0])))
plt.hlines(y=0 ,xmin=0,xmax=65)


# In[108]:





# In[174]:


plt.figure()
plt.plot(torch.mean(fullA[1:,run].float(),dim=(1,2)),'.')
plt.hlines(y = torch.mean(fullA[0,run].float()), xmin = 0 , xmax = 65)
plt.grid()


# In[180]:


plt.figure()
for i,direction in enumerate([33,34,44,60]):

    plt.errorbar(x= np.arange(5)+i*0.05,y=run0[run,:,:n_shots,direction].mean(1),yerr = run0[run,:,:n_shots,direction].std(1),label='direction '+str(direction),ls ='None')
    plt.grid()
plt.title('best direction to remove')
plt.xlabel('FS class')
plt.ylabel('value along direction')
plt.legend()

for i,direction in enumerate([33,34,44,60]):

    plt.errorbar(x= np.arange(5,10)+i*0.05,y=run0[run,:,n_shots:,direction].mean(1),yerr = run0[run,:,n_shots:,direction].std(1),label='direction '+str(direction),ls='None')
    plt.grid()
plt.title('best direction to remove')
plt.xlabel('FS class')
plt.ylabel('value along direction')
plt.legend()


# In[178]:


plt.figure()
for i,direction in enumerate([4,23,24,18,22]):

    plt.errorbar(x= np.arange(5)+i*0.05,y=run0[run,:,:n_shots,direction].mean(1),yerr = run0[run,:,:n_shots,direction].std(1),label='direction '+str(direction))
    plt.grid()

plt.xlabel('FS class')
plt.ylabel('value along direction')
plt.legend()
plt.title('worst direction to remove')

for i,direction in enumerate([4,23,24,18,22]):
    plt.errorbar(x= np.arange(5,10)+i*0.05,y=run0[run,:,n_shots:,direction].mean(1),yerr = run0[run,:,n_shots:,direction].std(1),label='direction '+str(direction))
    plt.grid()
plt.title('worst direction to remove')
plt.xlabel('FS class')
plt.ylabel('value along direction')
plt.legend()


# ### cosine distance with base class + EUCLIDIAN distance with other shots

# In[27]:


run=1
get_ipython().run_line_magic('matplotlib', 'qt5')
cd = cosine_distances(run0[run,:,:n_shots].reshape(-1,64),base64)
plt.figure()
plt.imshow(cd)
plt.colorbar()


cd = torch.cdist(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))
plt.figure()
plt.imshow(cd)
plt.colorbar()

plt.figure()
plt.plot(torch.mean(fullA[1:,run].float(),dim=(1,2)),'.')
plt.hlines(y = torch.mean(fullA[0,run].float()), xmin = 0 , xmax = 65)
plt.grid()


# ### cosine dist with base class and cosine dist between shots with a projection

# In[28]:


run=3

cd = cosine_distances(run0[run,:,:n_shots].reshape(-1,64),base64)
cd_shots = cosine_distances(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))

plt.figure()
plt.imshow(cd)
plt.colorbar()

plt.figure()
plt.imshow(cd_shots)
plt.colorbar()

plt.figure()
plt.plot(torch.mean(fullA[1:,run].float(),dim=(1,2)),'.')
plt.hlines(y = torch.mean(fullA[0,run].float()), xmin = 0 , xmax = 65)
plt.grid()


# In[29]:


proj = 36

run0_proj = run0.detach().clone()
run0_proj[:,:,:,proj]=0
cd_proj = cosine_distances(run0_proj[run,:,:n_shots].reshape(-1,64),run0_proj[run,:,:n_shots].reshape(-1,64))
cd = cosine_distances(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))

plt.figure()
plt.imshow(cd-cd_proj)
plt.colorbar()


# ### cosine dist with base class and EUCLIDIAN between shots with a projection

# In[30]:


run=7

cd = cosine_distances(run0[run,:,:n_shots].reshape(-1,64),base64)
cd_shots = torch.cdist(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))
plt.figure()
plt.imshow(cd_shots)
plt.colorbar()


plt.figure()
plt.imshow(cd)
plt.colorbar()


plt.figure()
plt.imshow(run0[run,:,:n_shots].reshape(-1,64))

plt.figure()
plt.plot(torch.mean(fullA[1:,run].float(),dim=(1,2)),'.')
plt.hlines(y = torch.mean(fullA[0,run].float()), xmin = 0 , xmax = 65)
plt.grid()


# In[31]:


proj = 59
run0_proj = run0.detach().clone()
run0_proj[:,:,:,proj]=0
cd_proj = torch.cdist(run0_proj[run,:,:n_shots].reshape(-1,64),run0_proj[run,:,:n_shots].reshape(-1,64))

diff =cd_shots-cd_proj
plt.figure()
plt.imshow(diff-diff.mean())
plt.colorbar()


# # cosine distances n_shots = 300

# In[26]:


logits = torch.einsum('cf,gsf->gsc',weightA,feat) + biasA
n_runs, batch_few_shot_runs = 20,10
n_ways=5
n_shots = 300
run_classes, run_indices = define_runs(n_ways, n_shots, 600-n_shots,20, [600 for i in range(20)])

for i in tqdm(range(65)):
    if i!=0:
        feature=logits.detach().clone()
        feature[:,:,i-1]=0
    else:
        feature =logits.detach().clone()
    A,meanA,varA = ncm(feature[:64], feature[-20:], run_classes, run_indices, n_shots)
    if i==0:
        fullA = A.unsqueeze(0)
        fullmeanA = meanA.unsqueeze(0)
        fullvarA = varA.unsqueeze(0)
    else:
        fullA = torch.cat((fullA, A.unsqueeze(0)) ,dim = 0)
        fullmeanA = torch.cat((fullmeanA, meanA.unsqueeze(0)) ,dim = 0)
        fullvarA = torch.cat((fullvarA, varA.unsqueeze(0)) ,dim = 0)
fullA300 = fullA.detach().clone()


# ## Analysis of variance

# In[39]:


run=0

get_ipython().run_line_magic('matplotlib', 'qt5')
plt.figure()
plt.plot(fullvarA[:,run,:n_ways])
plt.plot(fullvarA[:,run,-1],label='interclass variance')

plt.plot(torch.mean(fullA[:,run].float(),dim=(1,2))*4.5,label='accuracy')
plt.legend()


# In[45]:



plt.figure()
plt.plot(torch.mean(fullA[:,run].float(),dim=(1,2)),label='accuracy')
plt.legend()


# In[47]:


plt.figure()
plt.plot(fullvarA[:,run,:n_ways].mean(-1)-fullvarA[:,run,-1])


# In[46]:


plt.figure()
plt.plot(torch.mean(fullA[1:,run].float(),dim=(1,2)),'.')
plt.hlines(y = torch.mean(fullA[0,run].float()), xmin = 0 , xmax = 65)
plt.grid()


# In[33]:


print(run_indices.shape,run_classes.shape,logits.shape)
run0 = generate_runs(logits, run_classes, run_indices, 0)
base64 = torch.eye(base_mean.shape[0])


# In[34]:


run=0
plt.figure()
cd = cosine_distances(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))

plt.imshow(cd)
plt.colorbar()

plt.figure()
plt.plot(torch.mean(fullA[1:,run].float(),dim=(1,2)),'.')
plt.hlines(y = torch.mean(fullA[0,run].float()), xmin = 0 , xmax = 65)
plt.grid()


# In[35]:


proj = 61
run0_proj = run0.detach().clone()
run0_proj[:,:,:,proj]=0
cd_proj = torch.cdist(run0_proj[run,:,:n_shots].reshape(-1,64),run0_proj[run,:,:n_shots].reshape(-1,64))
cd = torch.cdist(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))
plt.figure()
plt.imshow(cd-cd_proj)
plt.title('proj effect')
plt.colorbar()

plt.figure()
plt.imshow((cd-cd_proj)*(1+cd))
plt.title('proj effect ratio')
plt.colorbar()


# In[36]:


def f(cd, cd_proj):
    return ((cd-cd_proj)*(1+cd)).sum()

proj1 = 61
proj2 = 19
proj3 = 31
proj4 = 51
run0_proj = run0.detach().clone()
run0_proj[:,:,:,proj1]=0
cd = torch.cdist(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))
cd_proj = torch.cdist(run0_proj[run,:,:n_shots].reshape(-1,64),run0_proj[run,:,:n_shots].reshape(-1,64))

best =  f(cd, cd_proj)

run0_proj = run0.detach().clone()
run0_proj[:,:,:,proj2]=0
cd = torch.cdist(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))
cd_proj = torch.cdist(run0_proj[run,:,:n_shots].reshape(-1,64),run0_proj[run,:,:n_shots].reshape(-1,64))

worst = f(cd, cd_proj)

run0_proj = run0.detach().clone()
run0_proj[:,:,:,proj3]=0
cd = torch.cdist(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))
cd_proj = torch.cdist(run0_proj[run,:,:n_shots].reshape(-1,64),run0_proj[run,:,:n_shots].reshape(-1,64))

best2 = f(cd, cd_proj)

run0_proj = run0.detach().clone()
run0_proj[:,:,:,proj4]=0
cd = torch.cdist(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))
cd_proj = torch.cdist(run0_proj[run,:,:n_shots].reshape(-1,64),run0_proj[run,:,:n_shots].reshape(-1,64))

worst2 = f(cd, cd_proj)

print(best-worst)
print(best2-worst2)


# In[37]:


def f(cd, cd_proj):
    return torch.nan_to_num(cd-cd.mean()/(1+cd_proj),nan=1).mean()

proj1 = 61
proj2 = 19
proj3 = 31
proj4 = 51
run0_proj = run0.detach().clone()
run0_proj[:,:,:,proj1]=0
cd = torch.cdist(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))
cd_proj = torch.cdist(run0_proj[run,:,:n_shots].reshape(-1,64),run0_proj[run,:,:n_shots].reshape(-1,64))

best =  f(cd, cd_proj)

run0_proj = run0.detach().clone()
run0_proj[:,:,:,proj2]=0
cd = torch.cdist(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))
cd_proj = torch.cdist(run0_proj[run,:,:n_shots].reshape(-1,64),run0_proj[run,:,:n_shots].reshape(-1,64))

worst = f(cd, cd_proj)

run0_proj = run0.detach().clone()
run0_proj[:,:,:,proj3]=0
cd = torch.cdist(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))
cd_proj = torch.cdist(run0_proj[run,:,:n_shots].reshape(-1,64),run0_proj[run,:,:n_shots].reshape(-1,64))

best2 = f(cd, cd_proj)

run0_proj = run0.detach().clone()
run0_proj[:,:,:,proj4]=0
cd = torch.cdist(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))
cd_proj = torch.cdist(run0_proj[run,:,:n_shots].reshape(-1,64),run0_proj[run,:,:n_shots].reshape(-1,64))

worst2 = f(cd, cd_proj)

print(best-worst)
print(best2-worst2)


# ## Centroide 300 shots

# In[38]:


run=5


plt.figure()
plt.plot(torch.mean(fullA[1:,run].float(),dim=(1,2)),'.')
plt.hlines(y = torch.mean(fullA[0,run].float()), xmin = 0 , xmax = 65)
plt.grid()
cd = torch.cdist(centroides,centroides)

plt.figure()
plt.imshow(cd)
plt.colorbar()
plt.title('original')


# In[ ]:



proj=40
centroides = run0[run,:,:300,:].mean(1)
centroides_proj = centroides.detach().clone()
centroides_proj[:,proj]=0
cd_proj = torch.cdist(centroides_proj,centroides_proj)


plt.figure()
plt.imshow(cd_proj-cd)
plt.colorbar()
plt.title('shift from proj '+ str(proj))


# In[156]:


cross_detector(cd_proj-cd,0.88)


# In[117]:





# In[135]:


def cross_detector(array,seuil):
    s=array.shape
    for i in range(s[0]):
        for j in range(s[1]):
            som = 0
            for k in range(0,i):
                som+=array[k,j]
            for k in range(i+1,s[0]):
                som+=array[k,j]
            for l in range(0,j):
                som+=array[i,l]
            for l in range(j+1,s[1]):
                som+=array[i,l]

            if som/array.sum()>seuil:
                return True
    return False
                
                
    


# In[162]:


def proj_cross(run):
    centroides = run0[run,:,:300,:].mean(1)
    
    cd_shots = torch.cdist(centroides,centroides)
    L=[]
    for i in range(64):
        centroides_proj = centroides.detach().clone()
        centroides_proj[:,i]=0
        cd_proj = torch.cdist(centroides_proj,centroides_proj)
        if cross_detector(cd_proj-cd_shots,0.9):
            L.append(i)

    return L


# In[ ]:





# In[168]:


proj_cross(run)


# ## Création indicateur de classe à supprimer 5shots

# In[ ]:


perfect_matrix = torch.ones((25,25)) 
perfect_matrix[:5,:5] = -1
perfect_matrix[5:10,5:10] = -1
perfect_matrix[10:15,10:15] = -1
perfect_matrix[15:20,15:20] = -1
perfect_matrix[20:25,20:25] = -1
plt.figure()
plt.imshow(perfect_matrix)
plt.colorbar()


# In[ ]:


def test_projs(run):
    cd_shots = torch.cdist(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))
    L=[]
    for i in range(64):
        run0_proj = run0[run].detach().clone()
        run0_proj[:,:,i]=0
        cd_proj = torch.cdist(run0_proj[:,:n_shots].reshape(-1,64),run0_proj[:,:n_shots].reshape(-1,64))
        score = ((cd_shots-cd_proj)*perfect_matrix).sum()
        L.append(score)
    L=np.array(L)
    print(L)
    return np.argsort(L)


# In[ ]:


logits = torch.einsum('cf,gsf->gsc',weightA,feat) + biasA
n_runs, batch_few_shot_runs = 20,10
n_ways=5
n_shots = 5
run_classes, run_indices = define_runs(n_ways, n_shots, 600-n_shots,20, [600 for i in range(20)])

for i in tqdm(range(65)):
    if i!=0:
        feature=logits.detach().clone()
        feature[:,:,i-1]=0
    else:
        feature =logits.detach().clone()
    A,meanA,varA = ncm(feature[:64], feature[-20:], run_classes, run_indices, n_shots)
    if i==0:
        fullA = A.unsqueeze(0)
        fullmeanA = meanA.unsqueeze(0)
        fullvarA = varA.unsqueeze(0)
    else:
        fullA = torch.cat((fullA, A.unsqueeze(0)) ,dim = 0)
        fullmeanA = torch.cat((fullmeanA, meanA.unsqueeze(0)) ,dim = 0)
        fullvarA = torch.cat((fullvarA, varA.unsqueeze(0)) ,dim = 0)
fullA5 = fullA.detach().clone()


# In[ ]:


print(run_indices.shape,run_classes.shape,logits.shape)
run0 = generate_runs(logits, run_classes, run_indices, 0)


# In[ ]:


run = 1
plt.figure()
plt.plot(torch.mean(fullA[1:,run].float(),dim=(1,2)),'.')
plt.hlines(y = torch.mean(fullA[0,run].float()), xmin = 0 , xmax = 65)
plt.grid()


# In[ ]:


test_projs(run)


# ### critère 2 Uniformité

# In[ ]:


perfect_matrix0 = torch.ones((25,25)) 
perfect_matrix0[:5,:5] = 0
perfect_matrix0[5:10,5:10] = 0
perfect_matrix0[10:15,10:15] = 0
perfect_matrix0[15:20,15:20] = 0
perfect_matrix0[20:25,20:25] = 0




def test_projs_2(run):
    cd_shots = torch.cdist(run0[run,:,:n_shots].reshape(-1,64),run0[run,:,:n_shots].reshape(-1,64))
    d_mean = (cd_shots*perfect_matrix0).mean()
    delta = perfect_matrix0*(cd_shots-d_mean)
    L=[]
    for i in range(64):
        run0_proj = run0[run].detach().clone()
        run0_proj[:,:,i]=0
        cd_proj = torch.cdist(run0_proj[:,:n_shots].reshape(-1,64),run0_proj[:,:n_shots].reshape(-1,64))
        score = ((cd_shots-cd_proj)*delta).sum()
        L.append(score)
    L=np.array(L)
    print(L)
    return np.argsort(L)


# In[ ]:





# In[ ]:





# In[ ]:




