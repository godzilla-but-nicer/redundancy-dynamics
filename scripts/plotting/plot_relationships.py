import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy
from scipy.stats import spearmanr

# load ipm synergy bias data
rules = pd.read_csv('data/k5/sampled_rules.csv', index_col=0)
ipm_full = pd.read_csv('data/k5/stats/ipm_synergy_bias.csv', index_col=0).reset_index()
ipm = ipm_full.merge(rules, on='rule')
ipm['rule'] = ipm['rule'].astype(int)

# load ipm pi atoms
ipm_singles = pd.read_csv('data/k5/pid/ipm.csv', index_col=0)
ipm_singles['rule'] = ipm_singles['rule'].astype(int)
ipm_sb = ipm.merge(ipm_singles, on = 'rule')

# load ipm pure synergy
ipm_ps = pd.read_csv('data/k5/stats/ipm_pure_synergy.csv', index_col=0)
ipm_ps['rule'] = ipm_ps['rule'].astype(int)
ipm_ps = ipm_ps.merge(rules, on = 'rule')

# load imin synergy bias data
rules = pd.read_csv('data/k5/sampled_rules.csv', index_col=0)
imin_full = pd.read_csv('data/k5/stats/imin_synergy_bias.csv', index_col=0).reset_index()
imin = imin_full.merge(rules, on='rule')
imin['rule'] = imin['rule'].astype(int)

# load imin pi atoms
imin_singles = pd.read_csv('data/k5/pid/imin.csv', index_col=0)
imin_singles['rule'] = imin_singles['rule'].astype(int)
imin_sb = imin.merge(imin_singles, on = 'rule')

# load imin pure synergy
imin_ps = pd.read_csv('data/k5/stats/imin_pure_synergy.csv', index_col=0)
imin_ps['rule'] = imin_ps['rule'].astype(int)
imin_ps = imin_ps.merge(rules, on = 'rule')

# calculate the rule entropies with binary encoding
def rule_to_ent(rule: int) -> float:
    n_digits = 2**5
    digits = []
    while True:
        if rule == 0:
            break
        else:
            digits.append(rule % 2)
            rule = np.floor(rule / 2)
    
    ons = np.sum(digits) / n_digits
    return entropy([ons, 1 - ons])

ipm['entropy'] = ipm['rule'].apply(lambda x: rule_to_ent(x))

# load canalization data
cana_full = pd.read_csv('data/k5/cana_sampled.csv', index_col=0)
clean_rules = []
for rule in cana_full['rule']:
    clean_rule = rule.replace('[', '')
    clean_rule = clean_rule.replace(']', '')
    clean_rules.append(int(clean_rule))
cana_full['rule'] = clean_rules

# input symmetry
cana_full['ks'] = cana_full['s(0)'] + cana_full['s(1)'] + cana_full['s(2)'] + cana_full['s(3)'] + cana_full['s(4)']
# normalized input symmetry
cana_full['ks*'] = cana_full['ks'] / 25

cana = cana_full.merge(rules, on='rule')
cana['ke'] = 1 - cana['kr*']
cana['ka'] = 1 - cana['ks*']

# load transfer entropy data
directed = pd.read_csv('data/k5/stats/directed.csv', index_col=0).replace(-1, np.nan)
ais = directed[['rule', 'ais']].dropna()
te = directed[['rule', 'te_0->', 'te_1->', 'te_3->', 'te_4->']].dropna()

# load O-information data
o_info = pd.read_csv('data/k5/stats/o_information_new.csv')
o_info = o_info[o_info['p'] < 0.05]

# ipm pure synergy
cana_ipm_ps = cana.merge(ipm_ps).dropna()
print('# of values w/ ke and ipm pure synergy:', cana_ipm_ps.shape[0])
print('pure synergy:', spearmanr(cana_ipm_ps['pure_synergy'], cana_ipm_ps['ke']))
plt.scatter(cana_ipm_ps['pure_synergy'], cana_ipm_ps['ke'])
plt.xlabel('Pure Synergy')
plt.ylabel(r'$k_e$')
plt.savefig('plots/k5/relationships/ipm_ps_ke.png')
plt.savefig('plots/k5/relationships/ipm_ps_ke.pdf')
plt.savefig('plots/k5/relationships/ipm_ps_ke.svg')

# imin pure synergy
cana_ipm_ps = cana.merge(imin_ps).dropna()
print('# of values w/ ke and ipm pure synergy:', cana_ipm_ps.shape[0])
print('pure synergy:', spearmanr(cana_ipm_ps['pure_synergy'], cana_ipm_ps['ke']))
plt.figure()
plt.scatter(cana_ipm_ps['pure_synergy'], cana_ipm_ps['ke'])
plt.xlabel('Pure Synergy')
plt.ylabel(r'$k_e$')
plt.savefig('plots/k5/relationships/imin_ps_ke.png')
plt.savefig('plots/k5/relationships/imin_ps_ke.pdf')
plt.savefig('plots/k5/relationships/imin_ps_ke.svg')