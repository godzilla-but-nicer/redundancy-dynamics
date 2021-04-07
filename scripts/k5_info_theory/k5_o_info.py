import dit
import pandas as pd
import numpy as np
from tqdm import tqdm
from casim.utils import to_binary
from casim.CA1D import CA1D

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

def get_neighborhood_probs(time_series, k):
    """ extract the probabilities associated with each possible neighborhood 
    configuration """
    vectors = []
    for i in range(time_series.shape[1] - k):
        vectors.append(time_series[:, i:i+k])
    
    vectors = np.vstack(vectors)

    unique, counts = np.unique(vectors, axis=0, return_counts=True)
    probs = counts / np.sum(counts)
    return unique, probs

# set random state
rng = np.random.default_rng(1234)

# vars for later
n_trial = 10
k_neighbors = 5
ca_cells = 1000
bootstrap_count = 500
n_inputs = 2**k_neighbors
df_dict = []
input_ordering = make_input_strings(k_neighbors)
for l in tqdm(range(1, 30)): # of course this needs to actually cover all relevant lambda values
    print('Lambda = ' + str(l))
    l_df = pd.read_csv('data/k5/attractors/lambda_' + str(l) + '_attractors.csv')
    for rule in np.unique(l_df['rule']):
        row = {}  # becomes dataframe row

        # first we have to estimate neighborhood probabilities 
        # I'll simulate a huge CA and use it to estimate 5-cell vector probs
        # because its allself similar? I think thats legit? Lizier did it
        ca = CA1D(k_neighbors, rule)
        init_state = rng.choice([0,1], size=ca_cells)
        ca.set_state(init_state)
        ca_run = ca.simulate_time_series(ca_cells, 530)
        neighborhoods, probs = get_neighborhood_probs(ca_run, k_neighbors)

        # we need to stringify the vectors to appease DIT
        neighbor_strs = [''.join(vec.astype(str)) for vec in neighborhoods]

        # need to format strings how dit likes it, will be list of strings
        # use dit to calculate the o information
        dist = dit.Distribution(neighbor_strs, probs)
        o_info = dit.multivariate.o_information(dist)

        # we'll also to a null model where we permute each ca_run and do the
        # same calculations
        null_infos = np.zeros(bootstrap_count)
        ca_run_flat = ca_run.flatten()  # need 1d array for np.permute
        for i in range(bootstrap_count):
            ca_run_permuted = rng.permutation(ca_run_flat).reshape(ca_run.shape)
            neighb, probs = get_neighborhood_probs(ca_run_permuted, k_neighbors)
            neighbor_strs = [''.join(vec.astype(str)) for vec in neighb]
        
            dist = dit.Distribution(neighbor_strs, probs)
            null_infos[i] = dit.multivariate.o_information(dist)
        
        p_val = np.sum(o_info > null_infos) / null_infos.shape[0] 
        # Update dictionary to contain row for this term
        print(o_info, p_val)
        row = {'rule': rule, 'o-information': o_info, 'p': p_val}
        df_dict.append(row)

df = pd.DataFrame(df_dict)
df_fout = open('data/k5/stats/o_information.csv', 'w')
df.to_csv(df_fout)
