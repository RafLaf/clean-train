from args import args
import numpy as np
import torch
import json
import os
from utils import *
import few_shot_eval
import datasets
from tqdm import tqdm
from datasets import file

if args.dataset != "" and  not os.path.exists(file):
    if args.episodic:
        args.episodic = False  #get datasert info to create the novel run
        loaders, input_shape, num_classes, few_shot, top_5 = datasets.get_dataset(args.dataset)
        args.episodic = True
    if args.dataset.lower() in ["tieredimagenet", "cubfs"]:
        elements_train, elements_val, elements_novel = num_classes[-1]
        run_classes, run_indices = few_shot_eval.define_runs(args.n_ways, args.n_shots[0], args.n_queries, num_classes[2], elements_novel, num_runs=args.runs) #+ num_classes[0]+ num_classes[1]
    else:
        elements_train = [num_classes[-1]]*num_classes[0]
        run_classes, run_indices = few_shot_eval.define_runs(args.n_ways, args.n_shots[0], args.n_queries, num_classes[2], [num_classes[-1]]*num_classes[2], num_runs=args.runs) 
    print('Novel_run = {} \n novel _indices {}'.format(run_classes, run_indices) )
    print(f'{run_classes.shape=}{run_indices.shape=}{run_indices.shape=}')




def load_features(num_classes):
    features = torch.load(args.features_epi, map_location = args.device)
    features = preprocess(features[:num_classes], features)
    length = features.shape[1]*num_classes
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



def get_cost(classes, L_indices, run = 0):
    dim = loaded_features.shape[-1]
    cclasses = torch.gather(loaded_features, 0, classes.reshape(-1,1,1).repeat(1,loaded_features.shape[1],dim))
    run_train = torch.gather(cclasses, 1, L_indices.unsqueeze(-1).repeat(1,1,dim))
    cclasses = torch.gather(loaded_features[num_classes[0]+num_classes[1]:], 0, run_classes.reshape(-1,1,1).repeat(1,loaded_features.shape[1],dim))
    run_test = torch.gather(cclasses, 1, run_indices[0].unsqueeze(-1).repeat(1,1,dim))
    _,cost =getProbas(run_train.reshape(1,-1,dim), run_test.reshape(1,-1,dim))
    return cost

def get_cost_stats(run=0):
    cost_list=[]
    for i in range(100):
        classes, L_indices = random_episode()
        cost = get_cost(classes, L_indices, run=run)
        cost_list.append(cost.item())
    cost_list = np.array(cost_list)
    avg_cost, std_cost =  cost_list.mean(), cost_list.std()
    return avg_cost, std_cost
    
def random_episode():
    classes = torch.randperm(num_classes[0])[:args.n_ways].to(args.device)
    n_samples = (episode_size // args.n_ways)
    L_indices = []
    for c in range(args.n_ways):
        class_indices = torch.randperm(elements_train[classes[c]])[:episode_size // args.n_ways].to(args.device)
        L_indices.append(class_indices)
    L_indices = torch.stack(L_indices)
    return classes , L_indices


def get_episode( idx , avg_cost, std_cost, run=0):
    iter = 0
    maxiter = 1000
    while True and iter< maxiter:
        iter+=1
        classes , L_indices = random_episode()
        cost = get_cost(classes, L_indices, run)
        if cost<avg_cost-0*std_cost:
            return classes, L_indices
    if iter==maxiter:
        raise ValueError('no good run found')


if not os.path.exists(file):
    episode_size = (args.batch_size // args.n_ways) * args.n_ways
    L_episodes = []
    loaded_features, length = load_features(num_classes[0])

    for i in range(args.runs):
        avg_cost, std_cost = get_cost_stats(run=i)
        episodes_cl, episodes_i = [],[]
        for idx in tqdm(range(args.epochs*args.episodes_per_epoch)):
            cl, ind = get_episode( idx , avg_cost, std_cost,run=i)
            episodes_cl.append(cl)
            episodes_i.append(ind)
        episodes_cl = np.array(episodes_cl)
        episodes_i = np.array(episodes_i)
    L_episodes =np.array(L_episodes)
    np.savez(file, classes = run_classes.cpu().detach().numpy(),indices_novel = run_classes.cpu().detach().numpy(), episodes_cl = episodes_cl, episodes_i = episodes_i)
    print('created and saved episodes for novel classes {}'.format(run_classes.cpu()))
    loaded_file = np.load(file)   
else:
    loaded_file = np.load(file)   
