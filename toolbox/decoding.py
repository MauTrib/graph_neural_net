import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform

'''
Set of techniques which allow to recover a tour in a graph given the edges 
prediction probabilities outputted by the GNNs
All the methods takes as input a batch of graphs heat maps of shape batch_size,n,n 
and use a method to output a tour.
Output shape is a list of edges of shape batch_size,n,2
'''
def is_permutation_matrix(x):
    '''
    Checks if x is a permutation matrix, thus a tour.
    '''
    x = x.squeeze()
    return (x.ndim == 2 and x.shape[0] == x.shape[1] and
            (x.sum(dim=0) == 1).all() and 
            (x.sum(dim=1) == 1).all() and
            ((x == 1) | (x == 0)).all())

def tour_to_perm(n,path):
    '''
    Transform a tour into a permutation (directed tour)
    '''
    m = torch.zeros((n,n))
    for k in range(n):
        u = int(path[k])
        v = int(path[(k+1)%n])
        m[u,v] = 1
    return m


def tour_to_adj(n,path):
    '''
    Transform a tour into an adjacency matrix
    '''
    m = torch.zeros((n,n))
    for k in range(n):
        u = int(path[k])
        v = int(path[(k+1)%n])
        m[u,v] = 1
        m[v,u] = 1
    return m

def greedy_decoding(G):
    '''
    Starts from the first node. At every steps, it looks for the most probable neighbors
    which hasn't been visited yet, which yields a tour at the end
    '''
    batch_size,n,_ = G.size()
    output = torch.zeros(batch_size,n,n)
    for k in range(batch_size):
        curr_output = torch.zeros(n)
        current = torch.randint(n,(1,1)).item()
        not_seen = torch.ones(n, dtype=torch.bool)
        not_seen[current] = False
        curr_output[0] = current
        counter = 1
        while counter < n:
            nxt = torch.argmax(G[k][current]*not_seen)
            not_seen[nxt] = False
            curr_output[counter] = nxt
            current = nxt
            counter+=1
            output[k] = tour_to_adj(n,curr_output)
    return output

def get_confused(n,G):
    """
    Gives the 'most-confused' node : the node that has the biggest std of probabilities
    Needs G.shape = n,n
    """
    maxi_std = -1
    node_idx = -1
    for node in range(n):
        cur_node = G[node,:]
        cur_std = cur_node.std()
        if cur_std>maxi_std:
            maxi_std = cur_std
            node_idx = node
    assert node_idx!=-1, "Should not be possible to have std always smaller than -1"
    return node_idx

def get_surest(n,G):
    """
    Gives the 'surest node : the node that has the biggest edge proba
    Needs G.shape = n,n
    """
    node_idx = torch.argmax(G.glatten())//n
    return node_idx

def beam_decode(raw_scores,xs,ys,b=5,start_mode="r",chosen=0):

    start_mode = start_mode.lower()
    if start_mode=='r':
        start_fn = lambda n, G : torch.randint(n,(1,1)).item()
    elif start_mode=='c':
        start_fn = lambda n, G : chosen
    elif start_mode=='conf': #Confusion
        start_fn = get_confused
    elif start_mode=='sure': #Start from the surest edge
        start_fn = get_surest
    else:
        raise KeyError, "Start function {} not implemented.".format(start_mode)

    
    with torch.no_grad(): #Make sure no gradient is computed
        G = torch.nn.Sigmoid()(raw_scores[0]).unsqueeze(0)

        bs,n,_ = G.shape
        
        output = torch.zeros(bs,n,n)

        diag_mask = torch.diag_embed(torch.ones(bs,n,dtype=torch.bool))
        G[diag_mask] = 0 #Make sure the probability of staying on a node is 0

        for k in range(bs):
            beams = torch.zeros(b,n, dtype=torch.int64)
            beams_score = torch.zeros((b,1))
            cur_g = G[k]
            start_node = start_fn(n,cur_g)
            cur_b = 1
            beams[:1,0] = start_node
            beams_score[:1] = 1
            for beam_time in range(1,n):
                not_seen = torch.ones((cur_b,n), dtype=torch.bool)
                not_seen.scatter_(1,beams[:cur_b,:beam_time],0) # Places False where a beam has already passed
                cur_neigh = cur_g[beams[:cur_b,beam_time-1]] #Love this syntax, just takes the neighbour values for each beam : cur_neigh.shape = (cur_b,n)
                nxt_values, nxt_indices = torch.topk(not_seen*cur_neigh,n,-1)
                nxt_values = nxt_values * beams_score[:cur_b]
                cur_b = min(b,cur_b*n)
                _, best_indices = torch.topk(nxt_values.flatten(), cur_b)
                best = torch.tensor(np.array(np.unravel_index(best_indices.numpy(), nxt_values.shape)).T)
                new_beams = torch.zeros(cur_b,n, dtype=torch.int64)
                for j in range(len(best)):
                    x,y = best[j]
                    new_beams[j,beam_time] = nxt_indices[x,y]
                    new_beams[j,:beam_time] = beams[x,:beam_time]
                    beams_score[j] = nxt_values[x,y]
                beams = new_beams
            #Now add last edge to the score
            beams_score = beams_score * torch.unsqueeze(cur_g[beams[:,-1],start_node],-1)
            
            nodes_coord = [ (xs[i],ys[i]) for i in range(len(xs))]
            W_dist = torch.tensor(squareform(pdist(nodes_coord, metric='euclidean')))

            mini = torch.sum(W_dist)
            best_beam_idx = -1
            for beam_num in range(beams.shape[0]):
                beam = beams[beam_num]
                path_length = 0
                for node in range(n):
                    path_length += W_dist[beam[node],beam[(node+1)%n]]
                if path_length<=mini:
                    mini=path_length
                    best_beam_idx = beam_num


            best_beam = beams[best_beam_idx]
            output[k] = tour_to_adj(n,best_beam)

            assert is_permutation_matrix(tour_to_perm(n,best_beam)), "Result of beam_fs is not a permutation !"
    
    return output












        



