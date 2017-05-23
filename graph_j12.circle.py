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
args = vars(parser.parse_args())

print igraph.__version__


relative = 0
file_idx = 0
edges_ref = []

for fileName in args['infiles']:
    file_idx += 1
    edges = np.loadtxt(fileName)
   
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

    communities = g.community_edge_betweenness(directed=False)
    clusters = communities.as_clustering()
   
    print clusters
    
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




