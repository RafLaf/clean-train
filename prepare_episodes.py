from args import args
import numpy as np
import torch
import json
import os
from utils import *
import few_shot_eval
import datasets
from tqdm import tqdm

file = 'data/episodes'+str(args.n_ways)+'ways'+str(args.batch_size)+'batch.npz'

if args.dataset != "" and  not os.path.exists(file):
    if args.custom_epi:
        args.episodic = False  #get datasert info to create the novel run
        loaders, input_shape, num_classes, few_shot, top_5 = datasets.get_dataset(args.dataset)
        args.episodic = True
    if args.dataset.lower() in ["tieredimagenet", "cubfs"]:
        elements_train, elements_val, elements_novel = num_classes[-1]
        run_classes, run_indices = few_shot_eval.define_runs(args.n_ways, args.n_shots[0], args.n_queries, num_classes[2], elements_novel) + num_classes[0]+ num_classes[1]
    else:
        run_classes, run_indices = few_shot_eval.define_runs(args.n_ways, args.n_shots[0], args.n_queries, num_classes[2], [num_classes[-1]]*num_classes[2]) 
        run_classes += num_classes[0]+ num_classes[1]
    print('Novel_run = {} \n novel _indices {}'.format(run_classes[0], run_indices[0]) )
    indices  = run_classes[0].unsqueeze(1)*num_classes[-1]  #find the solution tiered and cub
    indices_novel = run_indices[0] + indices




def load_features(num_classes):
    features = torch.load(args.test_features, map_location = args.device)
    features = preprocess(features[:num_classes], features)
    length = features.shape[1]*num_classes
    features = features.reshape(-1, features.shape[-1])
    return features,length




def compute_optimal_transport( M, r, c, epsilon=1e-6, lam= 0.01):
        r = r.to(args.device)
        c = c.to(args.device)
        n_runs, n, m = M.shape
        P = torch.exp(- lam * M)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)   
                                       
        u = torch.zeros(n_runs, n).to(args.device)
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)

            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        
        return P, torch.sum(P * M)
    
def getProbas(train_run, test_run):
    # compute squared dist to between samples
    n_runs, nb_samp_train , dim  = train_run.shape
    nb_samp_test = test_run.shape[1]
    dist = torch.cdist(train_run, test_run).pow(2)
        
    p_xj = torch.zeros_like(dist)
    r = torch.ones(n_runs, nb_samp_train)
    c = torch.ones(n_runs, nb_samp_test)
    p_xj, full_cost = compute_optimal_transport(dist, r, c, epsilon=1e-6)
    return p_xj, full_cost



def get_cost(L_indices):
    L_indices = np.array(L_indices).reshape(-1)
    run_train = loaded_features[L_indices]
    run_test = loaded_features[indices_novel]
    dim = run_train.shape[-1]
    _,cost =getProbas(run_train.reshape(1,-1,dim), run_test.reshape(1,-1,dim))
    return cost

def get_cost_stats(length):
    cost_list=[]
    for i in range(100):
        L_indices = random_episode()
        cost = get_cost(L_indices)
        cost_list.append(cost.item())
    cost_list = np.array(cost_list)
    avg_cost, std_cost =  cost_list.mean(), cost_list.std()
    return avg_cost, std_cost
    
def random_episode():
    classes = np.random.permutation(np.arange(num_classes[0]))[:args.n_ways]
    n_samples = (episode_size // args.n_ways)
    L_indices = []
    for c in range(args.n_ways):
        class_indices = np.random.permutation(np.arange(length // num_classes[0]))[:episode_size // args.n_ways]
        indices= (class_indices + classes[c] * (length // num_classes[0]))
        L_indices.append(indices)
    L_indices = np.array(L_indices)
    return L_indices


def get_episode( idx , avg_cost, std_cost,length):
    iter = 0
    maxiter = 1000
    while True and iter< maxiter:
        iter+=1
        L_indices = random_episode()
        cost = get_cost(L_indices)
        if cost<avg_cost-std_cost:
            index = np.stack(L_indices)
            return index
    if iter==maxiter:
        raise ValueError('no good run found')


if not os.path.exists(file):
    episode_size = (args.batch_size // args.n_ways) * args.n_ways
    loaded_features, length = load_features(num_classes[0])
    avg_cost, std_cost = get_cost_stats(length)

    episodes = []
    for idx in tqdm(range(args.epochs*episode_size)):
        episodes.append(get_episode( idx , avg_cost, std_cost,length))
    episodes_array = np.array(episodes)
    np.savez(file, classes = run_classes[0].cpu().detach().numpy(),indices_novel = indices_novel.cpu().detach().numpy(), episodes = episodes_array)
    print('created and saved episodes for novel classes {}'.format(run_classes[0].cpu()))
else:
    loaded_file = np.load(file)   
    print('loaded episodes for novel classes {}'.format(loaded_file['classes']))
