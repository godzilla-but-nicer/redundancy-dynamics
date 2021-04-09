import numpy as np
import pandas as pd
import sys
import time
from casim import CA1D

# unpack command-line args
N = int(sys.argv[1])
max_steps = N * 1000
lamb = int(sys.argv[2])
n_rules = int(sys.argv[3])
n_trials = int(sys.argv[4])

# these will become columns in long format dataframe
rule_list = []
measure_list = []
value_list = []
run_list = []

# initialize the thing
ca = CA1D.CA1D(5, 0, random_seed=666)

# initialize the simulation class
for rule in range(n_rules):
    # write to logs on bigred
    rule_start = time.time()
    # we will not get the same initial states in each rule but thats ok
    ca.set_rule(ca.lambda_rule(lamb))
    for trial in range(n_trials):
        init_state = ca.initialize_state(N, p=0.5)
        per, tra = ca.find_exact_attractor(N, max_steps, init_state)
        
        # add everything to the columns
        rule_list.append(ca.rule)
        measure_list.append('period')
        value_list.append(per)
        rule_list.append(ca.rule)
        measure_list.append('transient')
        value_list.append(tra)
        run_list.append(trial)
        run_list.append(trial)

    print('Rule ' + str(rule + 1) + '/' + str(n_rules) + ' complete in: ' + str(time.time() - rule_start))
    
# get everything into a dataframe to save a csv
df_dict = {'rule': rule_list, 'trial':run_list, 'measure': measure_list, 'value': value_list}
csv_path = 'data/k5/attractors/lambda_' + str(lamb) + '_attractors.csv'
df_out = pd.DataFrame(df_dict)
df_out.to_csv(csv_path)

# also save a text file with the params for the run
param_file = 'data/k5/attractors/lambda_' + str(lamb) + '_params.txt'
with open(param_file, 'w') as fout:
    fout.write('N: ' + str(N) + '\n')
    fout.write('max_steps: ' + str(max_steps) + '\n')
    fout.write('n_rules: ' + str(n_rules) + '\n')
    fout.write('n_trials: ' + str(n_trials) + '\n')

