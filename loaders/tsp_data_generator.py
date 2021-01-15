import os
import math
import random
import itertools
import networkx
import torch
import torch.utils
#from toolbox import utils
from concorde.tsp import TSPSolver
import timeit

# create directory if it does not exist
def check_dir(dir_path):
    dir_path = dir_path.replace('//','/')
    os.makedirs(dir_path, exist_ok=True)

GENERATOR_FUNCTIONS = {}

def dist_from_pos(pos):
    N = len(pos)
    W_dist = torch.zeros((N,N))
    for i in range(0,N-1):
        for j in range(i+1,N):
            curr_dist = math.sqrt( (pos[i][0]-pos[j][0])**2 + (pos[i][1]-pos[j][1])**2)
            W_dist[i,j] = curr_dist
            W_dist[j,i] = curr_dist
    return W_dist

def generates(name):
    """ Register a generator function for a graph distribution """
    def decorator(func):
        GENERATOR_FUNCTIONS[name] = func
        return func
    return decorator

@generates("GaussNormal")
def generate_gauss_normal_netx(N):
    """ Generate random graph with points"""
    pos = {i: (random.gauss(0, 1), random.gauss(0, 1)) for i in range(N)} #Define the positions of the points
    W_dist = dist_from_pos(pos)
    g = networkx.random_geometric_graph(N,0,pos=pos)
    g.add_edges_from(networkx.complete_graph(N).edges)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W_dist, dtype=torch.float)

@generates("Square01")
def generate_square_netx(N):
    pos = {i: (random.random(), random.random()) for i in range(N)} #Define the positions of the points
    W_dist = dist_from_pos(pos)
    g = networkx.random_geometric_graph(N,0,pos=pos)
    g.add_edges_from(networkx.complete_graph(N).edges)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W_dist, dtype=torch.float)


def is_swappable(g, u, v, s, t):
    """
    Check whether we can swap
    the edges u,v and s,t
    to get u,t and s,v
    """
    actual_edges = g.has_edge(u, v) and g.has_edge(s, t)
    no_self_loop = (u != t) and (s != v)
    no_parallel_edge = not (g.has_edge(u, t) or g.has_edge(s, v))
    return actual_edges and no_self_loop and no_parallel_edge

def adjacency_matrix_to_tensor_representation(W):
    """ Create a tensor B[:,:,1] = W and B[i,i,0] = deg(i)"""
    degrees = W.sum(1)
    B = torch.zeros((len(W), len(W), 2))
    B[:, :, 1] = W
    indices = torch.arange(len(W))
    B[indices, indices, 0] = degrees
    return B

def distance_matrix_tensor_representation(W):
    """ Create a tensor B[:,:,1] = W and B[i,i,0] = deg(i)"""
    W_adjacency = torch.sign(W)
    degrees = W_adjacency.sum(1)
    B = torch.zeros((len(W), len(W), 2))
    B[:, :, 1] = W
    indices = torch.arange(len(W))
    B[indices, indices, 0] = degrees
    return B


class Base_Generator(torch.utils.data.Dataset):
    def __init__(self, name, path_dataset, num_examples):
        self.path_dataset = path_dataset
        self.name = name
        self.num_examples = num_examples

    def load_dataset(self):
        """
        Look for required dataset in files and create it if
        it does not exist
        """
        filename = self.name + '.pkl'
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading dataset at {}'.format(path))
            data = torch.load(path)
            self.data = list(data)
        else:
            print('Creating dataset.')
            self.data = []
            self.create_dataset()
            print('Saving datatset at {}'.format(path))
            torch.save(self.data, path)
    
    def create_dataset(self):
        for i in range(self.num_examples):
            example = self.compute_example()
            self.data.append(example)

    def __getitem__(self, i):
        """ Fetch sample at index i """
        return self.data[i]

    def __len__(self):
        """ Get dataset length """
        return len(self.data)


class TSPGenerator(Base_Generator):
    """
    Build a numpy dataset of pairs of (Graph, noisy Graph)
    """
    def __init__(self, name, args):
        self.generative_model = args['generative_model']
        self.distance = args['distance_used']
        num_examples = args['num_examples_' + name]
        self.n_vertices = args['n_vertices']
        subfolder_name = 'TSP_{}_{}_{}_{}'.format(self.generative_model, 
                                                     self.distance,
                                                     num_examples,
                                                     self.n_vertices)
        path_dataset = os.path.join(args['path_dataset'],
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        
        
        check_dir(self.path_dataset)#utils.check_dir(self.path_dataset)

    def compute_example(self):
        """
        Compute pairs (Adjacency, Optimal Tour)
        """
        try:
            g, W = GENERATOR_FUNCTIONS[self.generative_model](self.n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        xs = [g.nodes[node]['pos'][0] for node in g.nodes]
        ys = [g.nodes[node]['pos'][1] for node in g.nodes]

        problem = TSPSolver.from_data(xs,ys,self.distance)
        solution = problem.solve(verbose=False)
        assert solution.success, "Couldn't find solution!"

        #print(W)
        B = distance_matrix_tensor_representation(W)
        
        SOL = torch.zeros((self.n_vertices,self.n_vertices),dtype=torch.int64)
        prec = solution.tour[-1]
        for i in range(self.n_vertices):
            curr = solution.tour[i]
            SOL[curr,prec] = 1
            SOL[prec,curr] = 1
            prec = curr

        return (B, SOL,(xs,ys))

    
if __name__=="__main__":
    name="train"
    args = {'generative_model': "GaussNormal",'num_examples_train':1,'n_vertices':5,'distance_used':'EUC_2D', 'path_dataset':"dataset_tsp"}
    tspg = TSPGenerator(name,args)
    time_taken = timeit.timeit(tspg.load_dataset,number=1)
    print(f"Took : {time_taken}s => {args['num_examples_train']/time_taken} TSPs per second")
    



