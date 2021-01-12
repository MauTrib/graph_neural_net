import os
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
    g = networkx.random_geometric_graph(N,0,pos=pos)
    g.add_edges_from(networkx.complete_graph(N).edges)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float)

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
        num_examples = args['num_examples_' + name]
        self.n_vertices = args['n_vertices']
        subfolder_name = 'TSP_{}_{}_{}'.format(self.generative_model,
                                                     num_examples,
                                                     self.n_vertices)
        path_dataset = os.path.join(args['path_dataset'],
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        
        
        check_dir(self.path_dataset)#utils.check_dir(self.path_dataset)

    def compute_example(self):
        """
        Compute pairs (Adjacency, Optimal Tour Distance)
        """
        try:
            g, W = GENERATOR_FUNCTIONS[self.generative_model](self.n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        xs = [g.nodes[node]['pos'][0] for node in g.nodes]
        ys = [g.nodes[node]['pos'][1] for node in g.nodes]

        problem = TSPSolver.from_data(xs,ys,"EUC_2D")
        solution = problem.solve(verbose=False)
        assert solution.success, "Couldn't find solution!"

        B = adjacency_matrix_to_tensor_representation(W)
        
        SOL = torch.zeros((self.n_vertices,self.n_vertices),dtype=torch.int64)
        prec = solution.tour[-1]
        for i in range(self.n_vertices):
            curr = solution.tour[i]
            SOL[curr,prec] = 1
            prec = curr

        return (B, SOL)

    
if __name__=="__main__":
    name="train"
    args = {'generative_model': "GaussNormal",'num_examples_train':200,'n_vertices':50,'path_dataset':"dataset_tsp"}
    tspg = TSPGenerator(name,args)
    time_taken = timeit.timeit(tspg.load_dataset,number=1)
    print(f"Took : {time_taken}s => {args['num_examples_train']/time_taken} TSPs per second")



