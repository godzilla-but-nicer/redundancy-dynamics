import pandas as pd
import numpy as np
import networkx as nx
import sys

# just so i dont make mistakes ill take a command line arg
method = sys.argv[1]

# data files
data_file = 'data/k5/pid/' + method + '.csv'  # 'rule' col and cols for all atoms
lattice_file = 'data/lattice_5.gml'

# root node for search
root = r'{0}{1}{2}{3}{4}'
tip = r'{0:1:2:3:4}'

# load data
info = pd.read_csv(data_file, index_col=0)
lattice = nx.read_gml(lattice_file)

# set up cols for calculations
info['E_syn'] = 0  
info['mutual_info'] = info.drop('rule', axis = 1).sum(axis = 1)

# set up the BFS
queue = [root]
layer = {root: 1}

# keep going til the queue is done
while queue:
    # remove the first element, thats what we check next
    base = queue.pop(0)

    # in excess synergy the bottom half of the lattice is negative
    if layer[base] < 16:
        coeff = -1
    elif layer[base] == 16:
        coeff = 0
    else:
        coeff = 1

    info['E_syn'] += info[base] * coeff

    for pred in [p for p in lattice.neighbors(base) if p not in layer]:
        queue.append(pred)
        # we know that this must be one step farther out than the checked node
        layer[pred] = layer[base] + 1


excess_synergy_df = info[['rule', 'E_syn', 'mutual_info']].copy()
notunq_levels = [layer[node] for node in layer.keys()]
unq_levels = np.unique(notunq_levels)
print(unq_levels)

width_at_layer = [node for node in layer.keys() if layer[node] == 16]
print(len(width_at_layer))
excess_synergy_df.to_csv('data/k5/stats/' + method + '_excess_synergy.csv', index = None)