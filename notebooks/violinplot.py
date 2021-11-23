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
import seaborn as sns
import pandas as pd


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
modelA = torch.load('modelA1',map_location=torch.device('cpu'))
biasA = modelA['linear.bias']
weightA = modelA['linear.weight']


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

# In[250]:


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
        return full_accuracy,full_mean,full_var,runs

    
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


# # cosine distances n_shots =5

# In[11]:


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


# In[12]:


plt.plot(torch.mean(fullA.float(), dim =(1,2,3)),'.')
plt.hlines(y = torch.mean(fullA[0].float()), xmin = 0 , xmax = 65)
plt.title('projection performance'+str(n_shots)+' shots' + str(n_runs) + ' runs')
plt.xlabel('projection')
plt.ylabel('performance')


# In[13]:


print(run_indices.shape,run_classes.shape,logits.shape)
run0 = generate_runs(logits, run_classes, run_indices, 0)
base64 = torch.eye(base_mean.shape[0])


# ### cosine distance with base class + cosine distance with other shots

# In[14]:


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


# In[15]:


plt.figure()
plt.plot(torch.mean(fullA[1:,run].float(),dim=(1,2)),'.')
plt.hlines(y = torch.mean(fullA[0,run].float()), xmin = 0 , xmax = 65)
plt.grid()


# In[16]:


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


# In[17]:


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


# In[18]:


get_ipython().run_line_magic('matplotlib', 'qt5')
fig, ax = plt.subplots()
run = 0
proj = 0 
vp = ax.violinplot(run0[run,:,:n_shots,proj], np.arange(5), widths=2,
                   showmeans=True, showmedians=False, showextrema=True)
vp2 = ax.violinplot(run0[run,:,n_shots:,proj], np.arange(5)+5, widths=2,
                   showmeans=True, showmedians=False, showextrema=True)



# ## Try to get somethin like below
# 

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_theme(style="whitegrid")
tips = sns.load_dataset("tips")
ax = sns.violinplot(x=tips["total_bill"])
ax = sns.violinplot(x="day", y="total_bill", hue='smoker',

                    data=tips, split=True ,hus_scale=True)


# We need a dataframe

# In[20]:


tips


# In[21]:


tips['total_bill'].dtype


# In[142]:


def get_data_frame(RUN,dim,n_shots=n_shots,n_ways=n_ways):
    dimRUN = RUN[:,:,dim].detach().clone()
    s = dimRUN.shape[1]
    dimRUN = dimRUN.reshape(-1)
    classe = torch.arange(n_ways).unsqueeze(0).repeat(s,1).T.reshape(-1)
    classe = classe.detach().numpy()
    support = np.array(['Yes' for i in range(n_shots)] +  ['No' for i in range(n_shots,600)])
    support = np.expand_dims(support,0)
    support = np.repeat(support,n_ways,axis=0).reshape(-1)
    df = pd.DataFrame(dimRUN, columns = [ 'value' ])
    df.insert(1 , 'class'  ,classe.astype('str'))
    df.insert(2 , 'support'  ,support)
    df['class'] = df['class'].astype('category')
    df['support'] =df['support'].astype('category')
    return df


# In[ ]:





# In[358]:


n_runs, batch_few_shot_runs =10,10
n_ways=2
n_shots = 1
run_classes, run_indices = define_runs(n_ways, n_shots, 15,20, [600 for i in range(20)])

for i in tqdm(range(65)):
    if i!=0:
        feature=logits.detach().clone()
        feature[:,:,i-1]=0
    else:
        
        feature =logits.detach().clone()
    A,meanA,varA,run0 = ncm(feature[:64], feature[-20:], run_classes, run_indices, n_shots)
    
    if i==0:
        fullA = A.unsqueeze(0)
        fullmeanA = meanA.unsqueeze(0)
        fullvarA = varA.unsqueeze(0)
    else:
        fullA = torch.cat((fullA, A.unsqueeze(0)) ,dim = 0)
        fullmeanA = torch.cat((fullmeanA, meanA.unsqueeze(0)) ,dim = 0)
        fullvarA = torch.cat((fullvarA, varA.unsqueeze(0)) ,dim = 0)
        run_old = run0.detach().clone()
    A,meanA,varA,run0 = ncm(logits[:64], logits[-20:], run_classes, run_indices, n_shots)
fullA5 = fullA.detach().clone()


# In[359]:


plt.figure()
plt.plot(torch.mean(fullA.float(),axis=(1,2,3)),'.')


# In[360]:


(torch.mean(fullA.float(),axis=(2,3)).max(0)[0] - fullA[0].float().mean()).mean()


# In[363]:


fullA[0].float().mean()


# In[371]:


get_ipython().run_line_magic('matplotlib', 'qt5')
run = 1

#run0 = generate_runs(logits, run_classes, run_indices, 0)

plt.figure()
plt.plot(torch.mean(fullA[1:,run].float(),dim=(1,2)),'.')
plt.hlines(y = torch.mean(fullA[0,run].float()), xmin = 0 , xmax = 65)
plt.xlabel('projection (0 is no projection)')
plt.ylabel('accuracy')
plt.grid()


# In[351]:


proj = 35
print('n_shots = ', n_shots)


df = get_data_frame(run0[run],proj ,n_shots=n_shots,n_ways=n_ways)
plt.figure()
ax = sns.violinplot(x="class", y="value", hue='support',data=df,scale_hue=True,split=True ,hus_scale=True, cut = 0)

boost = (torch.mean(fullA[proj+1,run].float()) - torch.mean(fullA[0,run].float())).item()
plt.title('direction ' + str(proj) + ' boost ' + str(np.round(boost,4)*100)+ '%')


# In[146]:


id_boost = torch.logical_and(torch.logical_not(fullA[0,run]) , fullA[proj+1,run] )
queries = run0[run,:,1:]


# In[148]:


#plt.figure()
plt.plot(queries[1,id_boost[1],proj],'.')
plt.ylabel('value along direction ' +str(proj))
plt.xlabel('samples')


# In[153]:


run0[run].shape


# In[229]:


def criterion_2shots(run):
    class_mean = run[:,:n_shots].mean(-2)
    query_mean = run[:,n_shots:].mean(-2)
    bad0 = abs(class_mean[0]-query_mean[1])- abs(class_mean[0]-query_mean[0])
    bad1 = abs(class_mean[1]-query_mean[0])- abs(class_mean[1]-query_mean[1])
    metric = bad0 + bad1
    return metric


# In[230]:


metric = criterion_2shots(run0[run])


# In[219]:


metric.argsort()


# In[224]:


boost = torch.mean(fullA[1:,run].float(),dim=(1,2)) - torch.mean(fullA[0,run].float()).item()
print(boost.argsort())


# In[231]:


for run in range(10):
    metric = criterion_2shots(run0[run])
    boost = torch.mean(fullA[1:,run].float(),dim=(1,2)) - torch.mean(fullA[0,run].float()).item()
    plt.figure()
    plt.plot(metric,boost,'.')

