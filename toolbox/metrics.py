import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.modules.activation import Sigmoid
from toolbox.utils import get_device

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

def accuracy_linear_assignment(weights,labels=None,aggregate_score = True):
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
        _ , preds = linear_sum_assignment(cost)
        if aggregate_score:
            acc += np.sum(preds == label)
            total_n_vertices += len(weight)
        else:
            all_acc += [np.sum(preds == label)/len(weight)]
        
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

    deg = torch.sum(weights, dim=-1)
    inds = [ (torch.topk(deg[k],int(clique_sizes[k].item()),dim=-1))[1] for k in range(bs)]
    for i,cur_w in enumerate(weights):
        sol = torch.sum(solutions[i],dim=1) #Sum over rows !
        ind = inds[i]
        for idx in ind:
            idx = idx.item()
            if sol[idx]:
                true_pos += 1
        total_n_vertices+=clique_sizes[i].item()
    """
    y_onehot = torch.zeros_like(weights)
    y_onehot.scatter_(2, inds, 1)
    true_pos = torch.sum(y_onehot==solutions)
    total_n_vertices = bs*clique_size"""
    return true_pos, total_n_vertices



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

def compute_f1_3(raw_scores,target):
    """
    Computes F1-score with the 3 best edges per row
    For TSP, the best result will be : prec=2/3, rec=1, f1=0.8 (only 2 edges are valid)
    """
    device = get_device(raw_scores)
    _, ind = torch.topk(raw_scores, 3, dim =2) #Here chooses the 3 best choices
    y_onehot = torch.zeros_like(raw_scores).to(device)
    y_onehot.scatter_(2, ind, 1)
    return f1_score(y_onehot,target)

def tsp_rl_loss(raw_scores, distance_matrix):
    proba = Sigmoid()(raw_scores)
    loss = torch.sum(torch.sum(proba*distance_matrix, dim=-1), dim=-1)
    return torch.mean(loss).data.item()

def accuracy_sbm(raw_scores,target):
    """
    Computes a simple category accuracy
    Needs raw_scores.shape = (bs,n) and target.shape = (bs,n)
    """
    bs,n= raw_scores.shape
    category = (raw_scores>0.5).to(int)
    true_pos = int(torch.sum(1-torch.abs(target-category)).cpu().item())
    return true_pos, bs * n 