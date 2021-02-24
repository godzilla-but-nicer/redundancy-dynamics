import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from casim.eca import eca_sim

# init rng
rng = np.random.default_rng(666)

rule = int(sys.argv[1])
N = int(sys.argv[2])
steps = N * 300

trials = int(sys.argv[3])
periods = np.zeros(trials)
transients = np.zeros(trials)

eca = eca_sim(rule)
for ti in tqdm(range(trials)):
    period, transient = eca.find_exact_attractor(
        N, steps, rng.choice([0, 1], size=N))
    periods[ti] = period
    transients[ti] = transient

df = pd.DataFrame({'period': periods, 'transient': transients})
df.to_csv('data/eca/attractors/rule_' + str(rule) + '/exact_attr_' + str(rule) + '_' + str(N) + '.csv')



