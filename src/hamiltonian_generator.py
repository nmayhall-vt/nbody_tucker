import numpy as np
import matplotlib 
#import igraph



def make_2d_lattice(size = (2,2),
                    blocks = [[0,1],[2,3]],
                    ratio = .5
                    ):
    """Create an mxn grid lattice"""
    n_blocks = len(blocks)
    n = size[0]
    m = size[1]



    n_sites = m*n
    intra_block_bonds = np.zeros((n_sites, n_sites)) 
    for bi in range(0,n_blocks):
        for si in blocks[bi]:
            for sj in blocks[bi]:
                intra_block_bonds[si,sj] = 1

    
    J = np.zeros((m*n,m*n))
    for i in range(0,m):
        for j in range(0,n):
            ij = j+i*n
            for k in range(0,m):
                for l in range(0,n):
                    kl = l+k*n
                    if abs(i-k) == 1 and j==l or abs(j-l) == 1 and i==k:
                        J[ij,kl] = -ratio
                        
                        if intra_block_bonds[ij,kl] == 1:
                            J[ij,kl] = -1

    #print(J)
    return(J)


def print_lattice(edges):
    
    print igraph.__version__
    
    relative = 0
    file_idx = 0
    edges_ref = []
    print edges
    if relative == 1:
        if file_idx == 1:
            edges_ref = edges
        else:
            edges = edges - edges_ref

    print edges
    N = edges.shape[1]
    g = igraph.Graph()
    g.add_vertices(N)
    weights = []
    labels = []
    for i in range(0,N):
        labels.append(i+1)
        for j in range(i+1,N):
            print i,j
            if abs(edges[i,j]) >= 1E-5:
                g.add_edge(i,j)
                weights.append(edges[i,j])

    print g
    edge_colors = []
    
    g.es["weight"] = 30*np.abs(weights)/np.max(np.abs(weights))
    for w in weights:
        wn = w/np.max(np.abs(weights))
        if w > 0:
            wn = 1
            c = "99,166,159,%3f" % wn
            edge_colors.append(c)
            #edge_colors.append('99,166,159,.7')
            #edge_colors.append('#63A69F')
        if w <= 0:
            wn = -1
            c = "242,131,107,%3f" % -wn
            edge_colors.append(c)
            #edge_colors.append('242,131,107,.7')
            #edge_colors.append('#F2594B')

    g.es["name"]   = labels 
    #g.es["weight"] = 20*np.abs(weights)

    #rt2 = np.sqrt(2)
    #rt2 = 3
    #l = [(0,0),(rt2,rt2),(7,rt2),(rt2+7,0),(14,0),(rt2+14,rt2),(21,rt2),(rt2+21,0)]

    layout = g.layout("circle")

    visual_style = {}
    visual_style["edge_curved"] = False 
    visual_style["vertex_size"] = 40
    visual_style["vertex_label"] = g.es["name"]
    visual_style["vertex_label_color"] = 'white'
    #visual_style["edge_width"] = 20 
    visual_style["edge_width"] = g.es['weight'] 
    visual_style["edge_color"] = edge_colors 
    visual_style["vertex_color"] = 'black'
    #visual_style["layout"] = l
    visual_style["layout"] = layout
    visual_style["bbox"] = (900, 900)
    visual_style["margin"] = 40
    #igraph.plot(g, **visual_style)
    igraph.plot(g,"%s.svg"%(fileName), **visual_style)
    #igraph.plot(g,"%s.pdf"%(fileName), **visual_style)




if __name__== "__main__":
    j12 = make_2d_lattice()
    
    #print_lattice(j12)
