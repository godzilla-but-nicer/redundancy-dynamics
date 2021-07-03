import dit
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from casim.utils import to_binary
from casim.CA1D import CA1D

def get_neighborhood_probs(stacked_neighborhoods):
    """ extract the probabilities associated with each possible neighborhood 
    configuration """

    unique, counts = np.unique(stacked_neighborhoods, axis=0, return_counts=True)

    unsorted_probs = counts / np.sum(counts)
    unsorted_string_vars = [''.join(uni.astype(str)) for uni in unique]

    vars, probs = (list(t) for t in zip(*sorted(zip(unsorted_string_vars, unsorted_probs))))

    return vars, probs

# set random state
rng = np.random.default_rng(1234)

# vars for later
bootstrap_count = 100
k_neighbors = 5
n_inputs = 2**k_neighbors
df_dict = []

for run_file in tqdm(glob('data/k5/runs/*80*.csv')):  
    row = {}  # becomes dataframe row

    # get the rule number
    rule = run_file.split('/')[-1].split('_')[0]
    row['rule'] = rule

    # load the neighborhood data. i might have to worry about the big files
    # but hopefully its just ok and i can just load them
    stacked_timeseries = np.loadtxt(run_file, delimiter=',')
    
    neighborhoods, probs = get_neighborhood_probs(stacked_timeseries)
    
    # need to format strings how dit likes it, will be list of strings
    # use dit to calculate the o information
    dist = dit.Distribution(neighborhoods, probs)
    o_info = dit.multivariate.o_information(dist)
    row['o-info'] = o_info
    
    # we'll also to a null model where we permute each ca_run and do the
    # same calculations
    null_infos = np.zeros(bootstrap_count)
    ts_flat = stacked_timeseries.flatten()  # need 1d array for np.permute
    for i in range(bootstrap_count):
        ts_permuted = rng.permutation(ts_flat).reshape(ts_flat.shape)
        neighb, probs = get_neighborhood_probs(ts_permuted)
    
        dist = dit.Distribution(neighb, probs)
        null_infos[i] = dit.multivariate.o_information(dist)
    
    # p value is fraction of times obs is greater or less than than null
    p_val = np.sum(o_info > null_infos) / null_infos.shape[0] 
    row['p'] = p_val

    # Update dictionary to contain row for this term
    df_dict.append(row)

df = pd.DataFrame(df_dict)
df_fout = open('data/k5/stats/o_information_new.csv', 'w')
df.to_csv(df_fout)
