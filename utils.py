from args import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st
import numpy as np

# function to display timer
def format_time(duration):
    duration = int(duration)
    s = duration % 60
    m = (duration // 60) % 60
    h = (duration // 3600)
    return "{:d}h{:02d}m{:02d}s".format(h,m,s)

def stats(scores, name):
    if len(scores) == 1:
        low, up = 0., 1.
    elif len(scores) < 30:
        low, up = st.t.interval(0.95, df = len(scores) - 1, loc = np.mean(scores), scale = st.sem(scores))
    else:
        low, up = st.norm.interval(0.95, loc = np.mean(scores), scale = st.sem(scores))
    print("{:s} {:.2f} (± {:.2f}) (conf: [{:.2f}, {:.2f}]) (worst: {:.2f}, best: {:.2f})".format(name, 100 * np.mean(scores), 100 * np.std(scores), 100 * low, 100 * up, 100 * np.min(scores), 100 * np.max(scores)))

class ncm_output(nn.Module):
    def __init__(self, indim, outdim):
        super(ncm_output, self).__init__()
        self.linear = nn.Linear(indim, outdim, bias = False)
        with torch.no_grad():
            self.linear.weight.data = self.linear.weight.data / torch.norm(self.linear.weight.data, dim = 1, p = 2, keepdim = True) * torch.mean(torch.norm(self.linear.weight.data, dim = 1, p = 2))
        self.linear = nn.utils.weight_norm(self.linear)
        self.temp = nn.Parameter(torch.zeros(1) - 1)

    def forward(self, x):
        x = x / torch.norm(x + 1e-6, dim = 1, p = 2, keepdim = True)
        return torch.norm(x.reshape(x.shape[0], 1, -1) - self.linear.weight_v.transpose(0,1).reshape(1, -1, x.shape[1]), dim = 2).pow(2) / self.temp

def linear(indim, outdim):
    if args.ncm_loss:
        return ncm_output(indim, outdim)
    else:
        return nn.Linear(indim, outdim)

def criterion_episodic(features, targets, n_shots = args.n_shots):
    feat = features.reshape(args.n_ways, -1, features.shape[1])
    feat = preprocess(feat)
    means = torch.mean(feat[:,:n_shots], dim = 1)
    dists = torch.norm(feat[:,n_shots:].unsqueeze(2) - means.unsqueeze(0).unsqueeze(0), dim = 3, p = 2).reshape(-1, args.n_ways).pow(2)
    return torch.nn.CrossEntropyLoss()(-1 * dists / args.temperature, targets.reshape(args.n_ways,-1)[:,n_shots:].reshape(-1))
    

def power(features):
    return torch.pow(features, 0.5)

def sphering(features):
    return features / torch.norm(features, p = 2, dim = 2, keepdim = True)

def centering(features):
    feat = features.reshape(-1, features.shape[2])
    feat = feat - feat.mean(dim = 0, keepdim = True).detach()
    features = feat.reshape(features.shape)
    return features

def preprocess(features):
    for i in range(len(args.preprocessing)):
        if args.preprocessing[i] == 'R':
            features = torch.relu(features)
        if args.preprocessing[i] == 'P':
            features = power(features)
        if args.preprocessing[i] == 'E':
            features = sphering(features)
        if args.preprocessing[i] == 'M':
            features = centering(features)
    return features
   
def proj_class(model,test_features):
    last_layer_weights=model.linear.weight                        #get classifier weight (last layer of resnet12)
    torch.save(last_layer_weights,'exp_proj/classifier' )
    for i in range (last_layer_weights.shape[0]):                 #one projection per 64 clesses on miniimagenet
        w=last_layer_weights[i]                                   #select weights of the i-th class
        proj = torch.matmul(test_features,w)/ torch.norm(w)**2    #get coef of projection and normalize
        projection_ortho = proj.unsqueeze(2).repeat(1,1,640)      
        projection_ortho = projection_ortho * w                   #vector of projection along w 
        projection_ortho = test_features - projection_ortho       #projection on the orthogonal space of w
        if i==0:
            full_projection_ortho=test_features.unsqueeze(0).to('cpu')                #save natural test features                            
            full_projection_ortho=torch.cat((full_projection_ortho,projection_ortho.unsqueeze(0).to('cpu')),dim=0) #add projections
        else:
            full_projection_ortho=torch.cat((full_projection_ortho,projection_ortho.unsqueeze(0).to('cpu')),dim=0) #add projections
    return full_projection_ortho