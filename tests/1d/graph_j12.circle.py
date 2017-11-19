#!/usr/bin/env python
import argparse,os,errno,re
import csv
import matplotlib 
import igraph
import numpy as np
#from colour import Color


#   Setup input arguments
parser = argparse.ArgumentParser(description='fill this out',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('infiles', nargs='+', help='List of input files to process')
parser.add_argument('-b','--blocks', type=str, default="heis_blocks.m", help='File containing vector of block sizes', required=False)
args = vars(parser.parse_args())

print igraph.__version__


"""
blocks = np.loadtxt(args['blocks']).astype(int)
n_blocks = len(blocks)
    
if len(blocks.shape) == 1:
    print 'blocks',blocks
    
    blocks.shape = (1,len(blocks.transpose()))
    n_blocks = len(blocks)

block_list = {}
for bi in range(0,n_blocks):
    block_list[str(blocks[bi,:])] = 1
    print str(blocks[bi,:])
print block_list
"""

relative = 0
file_idx = 0
edges_ref = []

for fileName in args['infiles']:
    file_idx += 1
    edges = np.loadtxt(fileName)
   
    if relative == 1:
        if file_idx == 1:
            edges_ref = edges
        else:
            edges = edges - edges_ref

    N = edges.shape[1]
    g = igraph.Graph()
    g.add_vertices(N)
    weights = []
    labels = []
    for i in range(0,N):
        labels.append(i+1)
        for j in range(i+1,N):
            if abs(edges[i,j]) >= 1E-5:

                # only add edges inside of blocks
                #todo
                g.add_edge(i,j)
                weights.append(edges[i,j])
            #weights.append(edges[i,j] * abs(edges[i,j]))
    
    print g
    #layout = g.layout("fr")
    #igraph.plot(g, layout = layout)

    #g.vs["label"] = g.vs["name"]
    #igraph.color_dict = {"m": "blue", "f": "pink"}
    #g.vs["color"] = [color_dict[gender] for gender in g.vs["gender"]]
    #igraph.plot(g, layout = layout, bbox = (300, 300), margin = 20)
    
    
    #visual_style["vertex_color"] = [color_dict[gender] for gender in g.vs["gender"]]
    #visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
    
    edge_colors = []
    
    #g.delete_edges(0,1)
    #g.delete_edges(2,3)
    layout = g.layout("grid", width=2)
    layout = g.layout("kk")
    layout = g.layout("fr")
    layout = g.layout("circle")
    
    #g.add_edge(0,1)
    #g.add_edge(2,3)
    
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

    visual_style = {}
    visual_style["edge_curved"] = False 
    visual_style["vertex_size"] = 40
    visual_style["vertex_label"] = g.es["name"]
    #visual_style["vertex_label_color"] = 'white'
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




