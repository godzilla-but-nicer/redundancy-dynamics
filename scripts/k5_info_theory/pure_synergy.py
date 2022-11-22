import numpy as np
import pandas as pd
import itertools
import sys

vars = (0, 1, 2, 3, 4)
def tuple_to_string(tup: tuple) -> str:
    """
    (0, 1) -> "{0:1}"
    (1, 2, 3, 4) -> "{1:2:3:4}"
    """
    ltup = [str(t) for t in tup]
    out_str = ':'.join(ltup)
    out_str = "{" + out_str + "}"

    return out_str

synergy_index = []
for k in (2, 3, 4):
    k_synergies = list(itertools.combinations(vars, k))
    for tup in k_synergies:
        synergy_index.append(tuple_to_string(tup))

# information measure
meas = sys.argv[1]
df = pd.read_csv('data/k5/pid/' + meas + '.csv', index_col=0)
df['sum'] = df.drop('rule', axis='columns').sum(axis='columns')

# keep 'rule', 'sum', and synergy indices
keep_cols = ['rule', 'sum']
keep_cols.extend(synergy_index)
syn = df[keep_cols]
syn['sum_pure_synergy'] = syn.drop(['rule', 'sum'], axis='columns').sum(axis='columns')
syn['pure_synergy'] = syn['sum_pure_synergy'] / syn['sum']
syn.to_csv('data/k5/stats/' + meas + '_pure_synergy.csv')
