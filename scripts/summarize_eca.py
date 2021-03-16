import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

N = int(sys.argv[1])

# load eca data or set up a path to load dynamics
data_path = 'data/eca/attractors/rule_{0}/approx_attr_{0}_{1}.csv'
class_df = pd.read_csv('data/eca/eca_equiv_classes.csv')

row_list = []

# plot for transient distributions
fig, ax = plt.subplots(2, 2)
ax = ax.flatten()

for cl in range(1, 5):
    cl_rules = class_df[class_df['wclass'] == cl]['rule']
    
    class_transients = []
    class_periods = []
    for rule in cl_rules:
        rule_df = pd.read_csv(data_path.format(rule, N)).dropna()

        # extract the transients and period we have to filter both based on 
        # false results indicated by huge transients
        transients = rule_df['transient']
        periods = rule_df['period']
        max_transient = np.round(0.90 * N*100)
        periods = periods[transients < max_transient]
        transients = transients[transients < max_transient]

        class_transients.extend(list(transients))
        class_periods.extend(list(periods))

        # get mean and variance of period
        row = {'rule' : rule,
               'class' : cl,
               'mean_transient' : np.mean(transients),
               'var_transient' : np.var(transients),
               'mean_period' : np.mean(periods),
               'var_period' : np.var(periods)}
        
        row_list.append(row)

        
    # filter inaccurate transients
    per = np.array(class_periods)
    trans = np.array(class_transients)

    # calculate histograms for transient and period
    n_bins = 20
    trans_hist, trans_edges = np.histogram(trans, bins=n_bins)
    per_hist, per_edges = np.histogram(per, bins=n_bins)

    ax[cl-1].plot(trans_edges[1:], trans_hist, label='Transient')
    ax[cl-1].set_title('Class' + str(cl))
    # ax[cl-1].plot(per_edges[1:], per_hist, label='Period')
    ax[cl-1].set_xlabel('Transient Lengh')
    ax[cl-1].set_ylabel('Count')

ax[1].legend()
plt.tight_layout()
plt.savefig('plots/eca/transient_hist_by_class.png')

out_df = pd.DataFrame(row_list)
out_df.to_csv('data/eca/attractors/eca_' + str(N) + '_summary.csv', index=False)