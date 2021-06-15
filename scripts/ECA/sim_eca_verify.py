import sys
import numpy as np
from tqdm import tqdm
from casim.eca import eca_sim

rule = int(sys.argv[1])
N = 16
steps = 500
trials = 1000

period = []
transients = []

exact_period = []
exact_transients = []

rng = np.random.default_rng(1234)

for t in tqdm(range(trials)):
    eca = eca_sim(rule, random_seed=1234, float_precision=16)
    
    # this is awkward that I have to generate states here
    # need to fix that
    state = rng.choice([0, 1], size=N)
    per, tra = eca.find_approx_attractor(N, steps, state)
    period.append(per)
    transients.append(tra)

    per, tra = eca.find_exact_attractor(N, steps, state)
    exact_period.append(per)
    exact_transients.append(tra)

data_path = 'data/eca/attractors/rule_' + str(rule) + '/'
with open(data_path + 'approx_16_periods_' + str(rule) + '.txt', 'w') as fout:
    for per in period:
        fout.write(str(per) + '\n')

with open(data_path + 'approx_16_transients_' + str(rule) + '.txt', 'w') as fout:
    for tra in transients:
        fout.write(str(tra) + '\n')

with open(data_path + 'exact_16_periods_' + str(rule) + '.txt', 'w') as fout:
    for per in exact_period:
        fout.write(str(per) + '\n')

with open(data_path + 'exact_16_transients_' + str(rule) + '.txt', 'w') as fout:
    for tra in exact_transients:
        fout.write(str(tra) + '\n')