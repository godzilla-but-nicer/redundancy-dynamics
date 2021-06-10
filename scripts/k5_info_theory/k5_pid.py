import dit
import networkx as nx
import numpy as np
import os
import pandas as pd
import sys
from glob import glob
from tqdm import tqdm
from casim.utils import to_binary

rule = sys.argv[1]

pid_method = sys.argv[2]

if pid_method == 'imin':
    pid = dit.pid.PID_WB(dist)
elif pid_method == 'ipm':
    pid = dit.pid.PID_PM(dist)
else:
    raise ValueError("method must be 'ipm' or 'imin'")


# assume only one data file per rule
data_file = glob('data/k5/runs/' + rule + '*.csv')[0]
time_series = np.loadtxt(data_file, delimiter = ',')

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
for key in pid_output._pis.keys():
    pis[str(key)] = pid_output._pis[key]
df_dict.append(pis)

df_out = pd.DataFrame(df_dict)
df_out.to_csv('data/k5/pid/' + rule + '_' + sys.argv[2] + '.csv')

