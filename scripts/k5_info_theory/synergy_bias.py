import pandas as pd
import numpy as np
import networkx as nx
import sys

# just so i dont make mistakes ill take a command line arg
method = sys.argv[1]

# data files
data_file = 'data/k5/pid/' + method + '_.csv'  # 'rule' col and cols for all atoms
lattice_file = 'data/lattice_5.gml'

# root node for search
root = r'{0}{1}{2}{3}{4}'
tip = r'{0:1:2:3:4}'

# load data
info = pd.read_csv(data_file)
info['B_syn'] = 0  # this gets values added to it downstream
lattice = nx.read_gml(lattice_file)

# set up the BFS
queue = [root]
layer = {root: 1}
# keep going til the queue is done
while queue:
    # remove the first element, thats what we check next
    base = queue.pop(0)

    # weigh the column by the layer -- will normalize at the end
    info['B_syn'] += info[base] * layer[base]

    for pred in [p for p in lattice.neighbors(base) if p not in layer]:
        queue.append(pred)
        # we know that this must be one step farther out than the checked node
        layer[pred] = layer[base] + 1

# normalize synergy bias
info['B_syn'] /= layer[tip]

synergy_bias_df = info[['rule', 'B_syn']].copy()

synergy_bias_df.to_csv('data/k5/stats/' + method + '_synergy_bias.csv', index = None)