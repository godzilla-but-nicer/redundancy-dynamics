import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.reshape.melt import wide_to_long
import seaborn as sns
from scipy.stats import entropy

# load ipm data
rules = pd.read_csv('data/k5/sampled_rules.csv', index_col=0)
ipm_full = pd.read_csv('data/k5/stats/ipm_synergy_bias.csv', index_col=0).reset_index()
ipm = ipm_full.merge(rules, on='rule')
ipm['rule'] = ipm['rule'].astype(int)

# load imin data
imin_full = pd.read_csv('data/k5/stats/imin_synergy_bias.csv', index_col=0).reset_index()
imin = imin_full.merge(rules, on = 'rule')
imin['rule'] - imin['rule'].astype(int)

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

# Load o-information data
o_info = pd.read_csv('data/k5/stats/o_information_new.csv')
o_info = o_info[o_info['p'] < 0.05]

# Load dynamics data
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

dyn = pd.DataFrame(dyn_rows)


### FIGURES ###
# Ipm Synergy Bias Histogram
plt.figure()
print('# Samples with valid I_pm B_syn: ', ipm.shape[0])
sns.histplot(ipm['B_syn'])
plt.xlabel(r'$B_{syn}$')
plt.savefig('plots/k5/distributions/ipm_bsyn_hist.png')
plt.savefig('plots/k5/distributions/ipm_bsyn_hist.pdf')
plt.savefig('plots/k5/distributions/ipm_bsyn_hist.svg')

# Imin Synergy Bias Histogram
plt.figure()
print('# Samples with valid I_min B_syn: ', imin.shape[0])
sns.histplot(imin['B_syn'])
plt.xlabel(r'$B_{syn}$')
plt.savefig('plots/k5/distributions/imin_bsyn_hist.png')
plt.savefig('plots/k5/distributions/imin_bsyn_hist.pdf')
plt.savefig('plots/k5/distributions/imin_bsyn_hist.svg')

# Effective Connectivity Histogram
plt.figure()
print('# Samples with valid k_e:', cana.shape[0])
sns.histplot(cana['ke'])
plt.xlabel(r'$k_e$')
plt.savefig('plots/k5/distributions/ke_hist.pdf')
plt.savefig('plots/k5/distributions/ke_hist.png')
plt.savefig('plots/k5/distributions/ke_hist.svg')

# Asymmetry Histogram
plt.figure()
print('# Samples with valid k_s:', cana.shape[0])
sns.histplot(cana['ks*'])
plt.xlabel(r'$k_s$')
plt.savefig('plots/k5/distributions/ks_hist.pdf')
plt.savefig('plots/k5/distributions/ks_hist.png')
plt.savefig('plots/k5/distributions/ks_hist.svg')


# Transfer Entropy histogram
plt.figure()
print('# Samples with transfer entropy:', te.shape[0])
te_long = pd.melt(te, id_vars='rule', value_vars=['te_0->', 'te_1->', 'te_3->', 'te_4->'])
sns.histplot(te_long['value'])
plt.xlabel('Transfer Entropy (bits)')
plt.savefig('plots/k5/distributions/te_hist.pdf')
plt.savefig('plots/k5/distributions/te_hist.png')
plt.savefig('plots/k5/distributions/te_hist.svg')

# Active Information Storage Histogram
plt.figure()
print('# AIS values', ais.shape[0])
sns.histplot(ais['ais'])
plt.xlabel('Active Information Storage (bits)')
plt.savefig('plots/k5/distributions/ais_hist.pdf')
plt.savefig('plots/k5/distributions/ais_hist.png')
plt.savefig('plots/k5/distributions/ais_hist.svg')

# O-information Histogram
plt.figure()
print('# O-information values', o_info.shape[0])
sns.histplot(o_info['o-info'])
plt.xlabel('O-information (bits)')
plt.savefig('plots/k5/distributions/o_info_hist.pdf')
plt.savefig('plots/k5/distributions/o_info_hist.png')
plt.savefig('plots/k5/distributions/o_info_hist.svg')


# Transient Length Histogram
plt.figure()
print('# Transients values', dyn.shape[0])
sns.histplot(dyn['mean_transient'], log_scale=(True, True))
plt.xlabel('Transient (timesteps)')
plt.savefig('plots/k5/distributions/trans_hist.pdf')
plt.savefig('plots/k5/distributions/trans_hist.png')
plt.savefig('plots/k5/distributions/trans_hist.svg')

# Period Histogram
plt.figure()
print('# Period values', dyn.shape[0])
sns.histplot(dyn['mean_period'], log_scale=(True, True))
plt.xlabel('Period (timesteps)')
plt.savefig('plots/k5/distributions/period_hist.pdf')
plt.savefig('plots/k5/distributions/period_hist.png')
plt.savefig('plots/k5/distributions/period_hist.svg')

# Output entropy
plt.figure()
print('# Rules with output entropy: {}'.format(len(ipm['entropy'].dropna())))
sns.histplot(ipm['entropy'])
plt.xlabel('Output Entropy (bit)')
plt.savefig('plots/k5/distributions/ent_hist.pdf')
plt.savefig('plots/k5/distributions/ent_hist.png')
plt.savefig('plots/k5/distributions/ent_hist.svg')