from casim.eca import eca_sim
import networkx as nx
import sys

rule = int(sys.argv[1])

eca = eca_sim(rule)
G = eca.get_state_transition_graph(16)

nx.write_edgelist(G, 'data/eca/stgs/rule_' + str(rule) + '.edgelist')
