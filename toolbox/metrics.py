import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.modules.activation import Sigmoid, Softmax
from toolbox.utils import get_device
import torch.nn.functional as F
from sklearn.cluster import KMeans

class Meter(object):
    """Computes and stores the sum, average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_sum(self):
        return self.sum
    
    def value(self):
        """ Returns the value over one epoch """
        return self.avg

    def is_active(self):
        return self.count > 0

class ValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def value(self):
        return self.val

def make_meter_loss():
    meters_dict = {
        'loss': Meter(),
        'loss_ref': Meter(),
        'batch_time': Meter(),
        'data_time': Meter(),
        'epoch_time': Meter(),
    }
    return meters_dict

def make_meter_acc():
    meters_dict = {
        'loss': Meter(),
        'acc': Meter(),
        'batch_time': Meter(),
        'data_time': Meter(),
        'epoch_time': Meter(),
    }
    return meters_dict

def make_meter_f1():
    meters_dict = {
        'loss': Meter(),
        'f1': Meter(),
        'precision': Meter(),
        'recall': Meter(),
        'batch_time': Meter(),
        'data_time': Meter(),
        'epoch_time': Meter(),
    }
    return meters_dict

#QAP

def accuracy_linear_assignment(weights, target, labels=None, aggregate_score=True):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    total_n_vertices = 0
    acc = 0
    all_acc = []
    for i, weight in enumerate(weights):
        if labels:
            label = labels[i]
        else:
            label = np.arange(len(weight))
        cost = -weight.cpu().detach().numpy()
        _, preds = linear_sum_assignment(cost)
        if aggregate_score:
            acc += np.sum(preds == label)
            total_n_vertices += len(weight)
        else:
            all_acc += [np.sum(preds == label) / len(weight)]

    if aggregate_score:
        return acc, total_n_vertices
    else:
        return all_acc

def all_losses_acc(val_loader,model,criterion,
            device,eval_score=None):
    model.eval()
    all_losses =[]
    all_acc = []

    for (input1, input2) in val_loader:
        input1 = input1.to(device)
        input2 = input2.to(device)
        output = model(input1,input2)

        loss = criterion(output)
        #print(output.shape)
        all_losses.append(loss.item())
    
        if eval_score is not None:
            acc = eval_score(output, aggregate_score=False)
            all_acc += acc
    return all_losses, np.array(all_acc)
   
def accuracy_max(weights,labels=None):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    acc = 0
    total_n_vertices = 0
    for i, weight in enumerate(weights):
        if labels is not None:
            label = labels[i]
        else:
            label = np.arange(len(weight))
        weight = weight.cpu().detach().numpy()
        preds = np.argmax(weight, 1)
        #print(preds)
        acc += np.sum(preds == label)
        total_n_vertices += len(weight)
    return acc, total_n_vertices

#MCP

def accuracy_max_mcp(weights,clique_size):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    true_pos = 0
    false_pos = 0
    total_n_vertices = 0
    for i, weight in enumerate(weights):
        weight = weight.cpu().detach().numpy()
        #print(weight)
        deg = np.sum(weight, 0)
        inds = np.argpartition(deg, -clique_size)[-clique_size:]
        #print(preds)#, np.sum(preds[:clique_size] <= clique_size))
        true_pos += np.sum(inds <= clique_size)
        #false_pos += np.sum(preds[clique_size:] < 0.1*clique_size)
        total_n_vertices += clique_size#len(weight)
    return true_pos, total_n_vertices

def accuracy_mcp(weights,solutions):
    """
    weights and solutions should be (bs,n,n)
    """
    clique_sizes,_ = torch.max(solutions.sum(dim=-1),dim=-1) #The '+1' is because the diagonal of the solutions is 0
    clique_sizes += 1
    bs,n,_ = weights.shape
    true_pos = 0
    total_n_vertices = 0

    probas = torch.sigmoid(weights)

    deg = torch.sum(probas, dim=-1)
    inds = [ (torch.topk(deg[k],int(clique_sizes[k].item()),dim=-1))[1] for k in range(bs)]
    for i,_ in enumerate(weights):
        sol = torch.sum(solutions[i],dim=1) #Sum over rows !
        ind = inds[i]
        for idx in ind:
            idx = idx.item()
            if sol[idx]:
                true_pos += 1
        total_n_vertices+=clique_sizes[i].item()
    return true_pos, total_n_vertices

#TSP

def f1_score(preds,labels):
    """
    take 2 adjacency matrices and compute precision, recall, f1_score for a tour
    """
    device = get_device(preds)

    labels = labels.to(device)
    bs, n_nodes ,_  = labels.shape
    true_pos = 0
    false_pos = 0
    mask = torch.ones((n_nodes,n_nodes))-torch.eye(n_nodes)
    mask = mask.to(device)
    for i in range(bs):
        true_pos += torch.sum(mask*preds[i,:,:]*labels[i,:,:]).cpu().item()
        false_pos += torch.sum(mask*preds[i,:,:]*(1-labels[i,:,:])).cpu().item()
        #pos += np.sum(preds[i][0,:] == labels[i][0,:])
        #pos += np.sum(preds[i][1,:] == labels[i][1,:])
    #prec = pos/2*n
    prec = true_pos/(true_pos+false_pos)
    rec = true_pos/(2*n_nodes*bs)
    if prec+rec == 0:
        f1 = 0.0
    else:
        f1 = 2*prec*rec/(prec+rec)
    return prec, rec, f1#, n, bs

def compute_f1(raw_scores,target,k_best=3):
    """
    Computes F1-score with the k_best best edges per row
    For TSP with the chosen 3 best, the best result will be : prec=2/3, rec=1, f1=0.8 (only 2 edges are valid)
    """
    device = get_device(raw_scores)
    _, ind = torch.topk(raw_scores, k_best, dim = 2) #Here chooses the 3 best choices
    y_onehot = torch.zeros_like(raw_scores).to(device)
    y_onehot.scatter_(2, ind, 1)
    return f1_score(y_onehot,target)

def tsp_rl_loss(raw_scores, distance_matrix):
    proba = Softmax(dim=-1)(raw_scores)
    proba = proba*proba.transpose(-2,-1)
    loss = torch.sum(torch.sum(proba*distance_matrix, dim=-1), dim=-1)
    return torch.mean(loss).data.item()

def tspd_dumb(raw_scores, target):
    """
    raw_scores and target of shape (bs,n,n)
    Just takes the order of the first column, it's a bit naive.
    As the data is ordered, it should be [0,1/n,2/n,...,1-1/n,1,1,1-1/n,...,1/n,0]
    The procedure is ordering the vertices and comparing directly
    """
    bs,n,_ = raw_scores.shape
    _,order = torch.topk(raw_scores,n,dim=2)
    results = order[:,0,:] #Keep the first row everytime
    true_pos=0
    for result in results:
        true_result = torch.cat( (result[::2],result[1::2].flip(-1)) )
        comparison = true_result==torch.arange(n)
        positives = torch.sum(comparison.to(int)).item()
        true_pos+=positives
    return true_pos,bs*n

#HHC
def accuracy_hhc(raw_scores, target):
    """ Computes simple accuracy by choosing the most probable edge
    For HHC:    - raw_scores and target of shape (bs,n,n)
                - target should be identity
     """
    bs,n,_ = raw_scores.shape
    device = get_device(raw_scores)
    _, ind = torch.topk(raw_scores, 1, dim = 2) #Here chooses the best choice
    y_onehot = torch.zeros_like(raw_scores).to(device)
    y_onehot.scatter_(2, ind, 1)
    accu = target*y_onehot #Places 1 where the values are the same
    true_pos = torch.count_nonzero(accu)
    n_total = bs * n #Perfect would be that we have the right permutation for every bs 
    return true_pos,n_total
    

#SBM
def accuracy_sbm_two_categories(raw_scores,target):
    """
    Computes a simple category accuracy
    Needs raw_scores.shape = (bs,n,out_features) and target.shape = (bs,n,n)
    """
    device = get_device(raw_scores)

    bs,n,_ = raw_scores.shape

    target_nodes = target[:,:,0] #Keep the category of the first node as 1, and the other at 0
    
    true_pos = 0

    embeddings = F.normalize(raw_scores,dim=-1) #Compute E
    similarity = embeddings @ embeddings.transpose(-2,-1) #Similarity = E@E.T
    for batch_embed,target_node in zip(similarity,target_nodes):
        kmeans = KMeans(n_clusters=2).fit(batch_embed.cpu().detach().numpy())
        labels = torch.tensor(kmeans.labels_).to(device)
        poss1 = torch.sum((labels==target_node).to(int))  
        poss2 = torch.sum(((1-labels)==target_node).to(int))
        best = max(poss1,poss2)
        #labels = 2*labels -1 #Normalize categories to 1 and -1
        #similarity = labels@labels.transpose(-2,-1)
        true_pos += int(best)
    return true_pos, bs * n

def accuracy_sbm_two_categories_edge(raw_scores,target):
    """
    Computes a simple category accuracy
    Needs raw_scores.shape = (bs,n,n) and target.shape = (bs,n,n)
    """
    device = get_device(raw_scores)

    bs,n,_ = raw_scores.shape

    #probas = torch.sigmoid(raw_scores) #No need for proba, we just need the best choices

    _,ind = torch.topk(raw_scores, n//2, -1)
    y_onehot = torch.zeros_like(raw_scores).to(device)
    y_onehot.scatter_(2, ind, 1)

    true_pos = bs * n * n - int(torch.sum(torch.abs(target-y_onehot)))
    return true_pos, bs * n * n