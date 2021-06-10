import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from tqdm import tqdm

rule = sys.argv[1]
data_path = "data/eca/attractors/rule_" + rule + "/"
etrans_data = data_path + "exact_16_transients_" + rule + ".txt"
atrans_data = data_path + 'approx_16_transients_' + rule + '.txt'
ptrans_data = data_path + 'stg_population_transients_' + rule + '.txt'

with open(etrans_data, 'r') as fin:
    etrans = fin.read().strip().split('\n')
    etrans = np.array([int(e) for e in etrans])

with open(atrans_data, 'r') as fin:
    atrans = fin.read().strip().split('\n')
    atrans = np.array([int(a) for a in atrans])

with open(ptrans_data, 'r') as fin:
    ptrans = fin.read().strip().split('\n')
    ptrans = [int(p) for p in ptrans]

rng = np.random.default_rng(1234)

strans = rng.choice(ptrans, size=len(atrans))

# compare exact and approximated values by simply plotting both lists
plt.figure()
plt.scatter(atrans, etrans)
plt.plot(range(min(atrans), max(atrans)), range(min(atrans), max(atrans)),
         c='grey')
plt.xlabel('Approximate Transients')
plt.ylabel('Matching Transients')
plt.savefig('plots/verify_approximation/' + rule + '_ea_plot.png')

# now we'll get into comparing the approximation with the STG derived "truth"
aquants = np.quantile(atrans, np.arange(0.01, 1, 0.01))
squants = np.quantile(strans, np.arange(0.01, 1, 0.01))
plt.figure()
plt.plot([1,atrans.max()], [1,strans.max()], c='grey')
plt.scatter(aquants, squants)
plt.xlabel('Approximation Quantiles')
plt.ylabel('STG Quantiles')
plt.savefig('plots/verify_approximation/' + rule + '_qq_plot.png')

equants = np.quantile(etrans, np.arange(0.01, 1, 0.01))
plt.figure()
plt.plot([1,etrans.max()], [1,strans.max()], c='grey')
plt.scatter(equants, squants)
plt.xlabel('Matching Quantiles')
plt.ylabel('STG Quantiles')
plt.savefig('plots/verify_approximation/' + rule + '_qq_plot_matching.png')

# we're also going to look at the difference in mean between permutations
# of the approximation transients and the STG transients
diff_true = np.mean(etrans) - np.mean(strans)
mixed = np.hstack((etrans, strans)) 

differences = np.zeros(50000)
for i in tqdm(range(differences.shape[0])):
    test = rng.choice(mixed, size=2000, replace=False)
    ta = test[:1000]
    ts = test[1000:]
    differences[i] = np.mean(ta) - np.mean(ts)

p_val = np.sum(diff_true > differences) / 50000

plt.figure()
plt.hist(differences)
plt.axvline(diff_true, c='C1', linestyle='--')
plt.xlabel('Permutated Difference of Means')
plt.ylabel('Count')
plt.title('p={:.3f}'.format(p_val))
plt.savefig('plots/verify_approximation/' + rule + '_perm_dist.png')
