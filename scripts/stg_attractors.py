import networkx as nx
import sys

rule = int(sys.argv[1])
edgelist = 'data/eca/stgs/rule_' + str(rule) + '.edgelist'
STG = nx.read_edgelist(edgelist, create_using=nx.DiGraph)

# number of attractors
n_attrs = nx.number_attracting_components(STG)

# we need to get a measure of period length for every state in the STG so that
# we can sample from it later
c_len = []
CC = [STG.subgraph(c).copy() for c in nx.weakly_connected_components(STG)]
for i, cc in enumerate(CC):
    period = len(next(nx.simple_cycles(cc)))
    c_len.extend([period]*len(cc))

# transient lengths
t_len = []

cycles = nx.simple_cycles(STG)
# need to do this for each attractor
for cycle in cycles:
    # set to ignore
    cyc_nodes = set(cycle)
    for node in cycle:
        # initialize the queue and our distance counter
        queue = [node]
        level = {node: 0}
        # keep going til the queue is done
        while queue:
            # remove the first element, thats what we check next
            base = queue.pop(0)
            for pred in [p for p in STG.predecessors(base) if p not in cyc_nodes]:
                queue.append(pred)
                # we know that this must be one step farther out than the checked node
                level[pred] = level[base] + 1
        t_len.extend([l for l in level.values()])


data_path = 'data/eca/attractors/rule_' + str(rule) + '/'
with open(data_path + 'n_attr_' + str(rule) + '.txt', 'w') as fout:
    fout.write(str(n_attrs) + '\n')

with open(data_path + 'stg_population_periods_' + str(rule) + '.txt', 'w') as fout:
    for cyc in c_len:
        fout.write(str(cyc) + '\n')

with open(data_path + 'stg_population_transients_' + str(rule) + '.txt', 'w') as fout:
    for tra in t_len:
        fout.write(str(tra) + '\n')
