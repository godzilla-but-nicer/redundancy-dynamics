import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import pandas as pd
from casim.CA1D import CA1D

# load dynamics dataframe for rule selection
raw_dyn = pd.read_csv('data/k5/combined_dynamics.csv', index_col=0)
dyn_rows = []

for rule in raw_dyn['rule'].unique():
    rule_rows = raw_dyn[raw_dyn['rule'] == rule]
    
    new_row = {}
    new_row['rule'] = int(rule)
    new_row['mean_transient'] = np.mean(rule_rows['transient'])
    new_row['se_transient'] = np.std(rule_rows['transient']) / np.sqrt(rule_rows.shape[0])
    new_row['min_obs_attr'] = len(rule_rows['period'].unique())
    new_row['mean_period'] = np.mean(rule_rows['period'])
    new_row['se_period'] = np.std(rule_rows['period']) / np.sqrt(rule_rows.shape[0])
    dyn_rows.append(new_row)

dyn_fixed = pd.DataFrame(dyn_rows)

# find rules with reasonable dynamics for illustration
transient_demo_rule = 2737453834

tra_sim = CA1D(5, transient_demo_rule, random_seed=420)
init_state = rng.choice(2, size=15)

# initialize the plot
fig, ax = plt.subplots(ncols = 2, figsize=(4, 4))

# find the transient and period then save the time series
tra_period, tra_transient = tra_sim.find_exact_attractor(15, 300, init_state)
n_timesteps = tra_transient + tra_period + round(0.6 * tra_period)
tra_sim.set_state(init_state)
time_series = tra_sim.simulate_time_series(15, n_timesteps)

# first figure
ax[0].fill_between(np.arange(31) - 0.5, tra_transient + 0.5, -0.5, alpha=0.2)
ax[0].fill_between(np.arange(31) - 0.5, tra_transient + tra_period + 0.5, tra_transient + 0.5, alpha=0.2)
ax[0].fill_between(np.arange(31) - 0.5, n_timesteps + 0.5, tra_transient + tra_period + 0.5, alpha=0.2)
ax[0].imshow(time_series, cmap='Greys')
ax[0].axhline(y=tra_transient + 0.4)
ax[0].axhline(y=tra_transient + 0.6, c='C1')
ax[0].axhline(y=tra_transient + tra_period + 0.4, c='C1')
ax[0].axhline(y=tra_transient + tra_period + 0.6, c='C2')
ax[0].set_yticks([tra_transient  + 0.5, tra_transient + tra_period + 0.5])
ax[0].set_yticklabels(['Transient', 'Period'],
                       rotation=90, va='bottom')
ax[0].set_xticks([])

# find the transient and period then save the time series
init_state = rng.choice(2, size=15)
tra_period, tra_transient = tra_sim.find_exact_attractor(15, 300, init_state)
tra_sim.set_state(init_state)
time_series = tra_sim.simulate_time_series(15, n_timesteps)

# second figure
ax[1].fill_between(np.arange(31) - 0.5, tra_transient + 0.5, -0.5, alpha=0.1)
ax[1].fill_between(np.arange(31) - 0.5, tra_transient + tra_period + 0.5, tra_transient + 0.5, alpha=0.1)
ax[1].fill_between(np.arange(31) - 0.5, n_timesteps + 0.5, tra_transient + tra_period + 0.5, alpha=0.1)
ax[1].imshow(time_series, cmap='Greys')
ax[1].axhline(y=tra_transient + 0.4, alpha=0.6)
ax[1].axhline(y=tra_transient + 0.6, c='C1', alpha=0.6)
ax[1].axhline(y=tra_transient + tra_period + 0.4, c='C1', alpha=0.6)
ax[1].axhline(y=tra_transient + tra_period + 0.6, c='C2', alpha=0.6)
#ax[1].set_yticks([tra_transient  + 0.5, tra_transient + tra_period + 0.5])
#ax[1].set_yticklabels(['Transient', 'First Attractor Cycle'],
#                      rotation=90, va='bottom')
ax[1].set_yticks([])
ax[1].set_xticks([])

plt.tight_layout()
plt.savefig('plots/k5/examples/example_ca.pdf')
plt.savefig('plots/k5/examples/example_ca.png', dpi=300)
plt.savefig('plots/k5/examples/example_ca.svg')
