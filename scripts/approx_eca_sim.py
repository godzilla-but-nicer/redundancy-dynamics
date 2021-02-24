import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from casim.eca import eca_sim

# init rng
rng = np.random.default_rng(666)

rule = int(sys.argv[1])
N = int(sys.argv[2])
steps = N * 100
word_size = max([3, int(N / 100 + 1)])

trials = int(sys.argv[3])
periods = np.zeros(trials)
transients = np.zeros(trials)

eca = eca_sim(rule)
for ti in tqdm(range(trials)):
    period, transient = eca.find_approx_attractor(
        N, steps, rng.choice([0, 1], size=N), block_size=5)
    periods[ti] = period
    transients[ti] = transient

df = pd.DataFrame({'period': periods, 'transient': transients})
df.to_csv('data/eca/attractors/rule_' + str(rule) + '/approx_attr_' + str(rule) + '_' + str(N) + '.csv')
np.savetxt('timeseries.txt', eca.entropies)
import matplotlib.pyplot as plt

plt.plot(eca.entropies)
plt.axvline(eca.approx_transient, c='C1')
plt.savefig('timeseries.png')


