import dit
import pandas as pd
import numpy as np
import sys
from glob import glob
from tqdm import tqdm
from casim.utils import to_binary

pid_method = sys.argv[1]
rule = sys.argv[2]

if pid_method == 'imin':
    pid = dit.pid.PID_WB(dist)
elif pid_method == 'pm':
    pid = dit.pid.PID_PM(dist)
else:
    raise ValueError("method must be 'ipm' or 'imin'")

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

# assumes that there is only one data file per rule
data_file = glob('data/k5/runs/' + rule + '*.csv')[0]
time_series = np.loadtxt(data_file)

# build up an array of all of the neigborhoods
wolfram_neighborhoods = np.zeros((2**5, 5))
for i in range(2**5):
    wolfram_neighborhoods[i,:] = to_binary(i, 2**5)


# vars for later
n_inputs = 2**5
df_dict = []
input_ordering = make_input_strings(5)
for l in range(2, n_inputs - 1):
    print('Lambda = ' + str(l))
    l_df = pd.read_csv('data/k5/attractors/lambda_' + str(l) + '_attractors.csv')
    for rule in tqdm(np.unique(l_df['rule'])):
        pis = {}  # gets filled with each pi term for this rule
        
        # need to format strings how dit likes it, will be list of strings
        arr = make_dist_arr(rule, input_ordering, len(input_ordering))
        print('doing the decomp now')

        # use dit to calculate the PID
        dist = dit.Distribution(arr, [1/n_inputs]*n_inputs)

        # Update dictionary to contain row for this term
        pis['rule'] = rule
        for key in pid._pis.keys():
            pis[str(key)] = pid._pis[key]
        df_dict.append(pis)

df = pd.DataFrame(df_dict)
df_fout = open('data/k5/stats/k5_pm.csv', 'w')
df.to_csv(df_fout)
