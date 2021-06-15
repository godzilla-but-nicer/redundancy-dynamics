import dit
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

# this function will build my distribution for 
def get_neighborhood_probs(stacked_neighborhoods):
    """ extract the probabilities associated with each possible neighborhood 
    configuration """

    unique, counts = np.unique(stacked_neighborhoods, axis=0, return_counts=True)

    unsorted_probs = counts / np.sum(counts)
    unsorted_string_vars = [''.join(uni.astype(str)) for uni in unique]

    vars, probs = (list(t) for t in zip(*sorted(zip(unsorted_string_vars, unsorted_probs))))

    return vars, probs

rng = np.random.default_rng(1234)

# vars for later
bootstrap_count = 100
df_dict = []

# files
run_files = glob('data/eca/runs/*')
rules = [f.split('/')[-1].split('_')[0] for f in run_files]

for rule in rules:
    stacked_timeseries = np.loadtxt(
                         'data/eca/runs/' + rule + '_500_300_100.csv',
                         delimiter=',')[:200000]

    row = {}  # becomes dataframe row

    # get joint probabilities
    states, probs = get_neighborhood_probs(stacked_timeseries)

    # need to format strings how dit likes it, will be list of strings
    # use dit to calculate the o information
    dist = dit.Distribution(states, probs)
    o_info = dit.multivariate.o_information(dist)

    # we'll also to a null model where we permute each ca_run and do the
    # same calculations
    flat_timeseries = stacked_timeseries.flatten()
    null_infos = np.zeros(bootstrap_count)

    for i in range(bootstrap_count):
        null_timeseries = np.reshape(rng.permutation(flat_timeseries), (-1, 5))

        null_states, null_probs = get_neighborhood_probs(null_timeseries)
        
        dist = dit.Distribution(null_states, null_probs)
        null_infos[i] = dit.multivariate.o_information(dist)
    
    p_val = np.sum(o_info > null_infos) / null_infos.shape[0] 
    # Update dictionary to contain row for this term
    print(rule, o_info, p_val)
    row = {'rule': rule, 'o-information': o_info, 'p': p_val}
    df_dict.append(row)

df = pd.DataFrame(df_dict)
df_fout = open('data/eca/stats/o_info.csv', 'w')
df.to_csv(df_fout)
