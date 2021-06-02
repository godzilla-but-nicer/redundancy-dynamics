import dit
import networkx as nx
import numpy as np
import os
import pandas as pd
import sys
from casim.utils import to_binary
from glob import glob

rule = sys.argv[1]

if sys.argv[2] == 'imin':
    pid_func = dit.pid.PID_WB
elif sys.argv[2] == 'ipm':
    pid_func = dit.pid.PID_PM
else:
    raise ValueError("PID method must be 'imin' or 'ipm'")

# these functions used to live in their own script, I dont have a good place for the mnow
# convert a number of inputs into the appropriate sets of inputs
# formats the inputs as strings of 1s and zeros in the order that wolframs rules are formatted
def make_input_strings(n_inputs):
    # set up variables for later
    input_sets = range(2**n_inputs)
    inputs = []

    # iterate over each numbered input set
    for inp in input_sets:
        # wolfram style input sets
        binary_input = to_binary(inp, n_inputs).astype(str)
        input_string = ''.join(binary_input)

        inputs.append(input_string)

    return inputs

# make a list of strings to pass to dit with the output on each string
def make_dist_arr(n, inputs, digits=8):
    outputs = to_binary(n, digits)
    dist = []
    for i, inp in enumerate(inputs):
        dist.append(inp + str(outputs[i]))
    return dist

# vars for later
n_inputs = 2**5
df_dict = []

# assume only one data file per rule
data_file = glob('data/k5/runs/' + rule + '*.csv')[0]
time_series = np.loadtxt(data_file, delimiter = ',')[:50]

# now we have to get the probs of each transition actually occuring
transitions = to_binary(int(rule), 2**5)
probs = np.zeros(2**5)
arr_strings = []

# to make searching faster later
enc_vec = 2 ** (np.arange(5, 0, -1) - 1)
encoded = np.dot(time_series, enc_vec)
total_obs = encoded.shape[0]
for neighborhood in range(2**5):
    # find every occurance of the neighborhood
    occurances = np.sum(encoded == neighborhood)
    probs[neighborhood] = occurances / total_obs
    print(probs[neighborhood])

    # now we need to build the strings for `dit`
    string_array = list(to_binary(neighborhood, 5).astype(str))
    string_array.append(str(transitions[neighborhood]))
    arr_strings.append(''.join(string_array))

print(np.sum(probs))

# use dit to calculate the PID
dist = dit.Distribution(arr_strings, probs)
pid_output = pid_func(dist)

# pull out the values of the pid
pis = {}
pis['rule'] = rule
for key in pid._pis.keys():
    pis[str(key)] = pid._pis[key]
df_dict.append(pis)

df_out = pd.DataFrame(df_dict)
df_out.to_csv('data/k5/pid/' + rule + '_' + sys.argv[2] + '.csv')

# if the lattice is not already saved in a format networkx can load than save it
if not os.path.exists('data/k5/pid/lattice.graphml'):
    nx.write_graphml(pid_output._lattice, 'data/k5/pid/lattice.graphml')