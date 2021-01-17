import networkx
import matplotlib.pyplot as plt
import os

def compare(xs,ys,adj_out,adj_target,save_out=True,path='pics/',name='graph'):
    N = len(xs)
    pos = {i: (xs[i], ys[i]) for i in range(N)}
    g = networkx.random_geometric_graph(N,0,pos=pos)
    g_out = networkx.random_geometric_graph(N,0,pos=pos)
    for i in range(N):
        for j in range(N):
            if 1==adj_out[i,j]==adj_target[i,j]:
                g.add_edge(i,j,color="black")
                g_out.add_edge(i,j,color="black")
            elif adj_out[i,j]:
                g.add_edge(i,j,color="red")
                g_out.add_edge(i,j,color="red")
            elif adj_target[i,j]:
                g.add_edge(i,j,color="blue")
    
    edges = g.edges()
    colors = [g[u][v]['color'] for u,v in edges]
    plt.figure()
    networkx.draw(g,pos,edge_color = colors)
    plt.tight_layout()
    plt.show()
    fname = os.path.join(path,name)
    plt.savefig(fname+".png", format="PNG")
    
    if save_out:
        plt.figure()
        edges = g_out.edges()
        colors = [g_out[u][v]['color'] for u,v in edges]
        networkx.draw(g_out,pos,edge_color = colors)
        plt.tight_layout()
        plt.show()
        fname = os.path.join(path,name)
        plt.savefig(fname+"_out.png", format="PNG")

def show_tour(xs,ys,adj,path='pics/',name='graph'):
    N = len(xs)
    pos = {i: (xs[i], ys[i]) for i in range(N)}
    g = networkx.random_geometric_graph(N,0,pos=pos)
    for i in range(N):
        for j in range(N):
            if adj[i,j]==1:
                g.add_edge(i,j,color="black")
    edges = g.edges()
    colors = [g[u][v]['color'] for u,v in edges]
    plt.figure()
    networkx.draw(g,pos,edge_color = colors)
    plt.tight_layout()
    plt.show()
    fname = os.path.join(path,name)
    plt.savefig(fname+".png", format="PNG")
