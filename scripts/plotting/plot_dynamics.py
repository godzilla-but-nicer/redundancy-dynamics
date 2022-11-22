import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.reshape.melt import wide_to_long
from scipy.stats.stats import spearmanr
import seaborn as sns
from scipy.stats import entropy


# load ipm synergy bias data
rules = pd.read_csv('data/k5/sampled_rules.csv', index_col=0)
ipm_full = pd.read_csv('data/k5/stats/ipm_synergy_bias.csv', index_col=0).reset_index()
ipm = ipm_full.merge(rules, on='rule')
ipm['rule'] = ipm['rule'].astype(int)

# load ipm synergy bias
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

# load imin lattice data
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

dyn_fixed = pd.DataFrame(dyn_rows)

#print(dyn_fixed.head())
#print(ipm_sb.head())


# Output entropy and dynamics
plt.figure()
dyn = dyn_fixed.copy()
dyn = dyn.merge(ipm, on='rule')
print('# Rules with entropy:', dyn.shape[0])
ent_vals = sorted(np.unique(dyn['entropy'].values))
se_periods = []
periods = []
se_transients = []
transients = []
for l in ent_vals:
    ld = dyn[dyn['entropy'] == l]
    periods.append(np.mean(ld['mean_period'].dropna()))
    se_periods.append(np.std(ld['mean_period'].dropna() / np.sqrt(len(ld['mean_period']))))
    transients.append(np.mean(ld['mean_transient'].dropna()))
    se_transients.append(np.std(ld['mean_transient'].dropna() / np.sqrt(len(ld['mean_transient']))))

# convert all to numpy arrays for easy math later
se_periods = np.array(se_periods)
periods = np.array(periods)
se_transients = np.array(se_transients)
transients = np.array(transients)

# entropy and transient
print('entropy v. transient:', spearmanr(ent_vals, transients))
print('entropy v. period:', spearmanr(ent_vals, periods))
plt.figure(figsize=(4,4))
plt.plot(ent_vals, periods, label='Period', marker='^', mfc='white', mec='C0')
plt.fill_between(ent_vals, periods - se_periods, periods + se_periods, color='C0', alpha = 0.4)
plt.plot(ent_vals, transients, label='Transient', marker='s', mfc='white', mec='C1')
plt.fill_between(ent_vals, transients - se_transients, transients + se_transients, color='C1', alpha = 0.4)
plt.xlabel(r'$H_{out}$')
plt.ylabel(r'Timesteps')
plt.ylim((1, 10**4))
plt.legend(loc='upper left')
plt.yscale('log')
plt.tight_layout()
plt.savefig('plots/k5/dynamics/entropy_dynamics.pdf')
plt.savefig('plots/k5/dynamics/entropy_dynamics.svg')
plt.savefig('plots/k5/dynamics/entropy_dynamics.png')


# Scatter plot of transient vs period
fig, ax = plt.subplots()
print('period v. transient:', spearmanr(dyn['mean_transient'], dyn['mean_period']))
ax.scatter(dyn['mean_transient'], dyn['mean_period'])
ax.set_xlabel('Transient')
ax.set_ylabel('Period')
ax.set_xscale('log')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('plots/k5/dynamics/dynamics_only.png')
plt.savefig('plots/k5/dynamics/dynamics_only.pdf')
plt.savefig('plots/k5/dynamics/dynamics_only.svg')

print('-'*40)
print('Info quantities against transient length')
print('-'*40)

# Transfer entropy and transient length
te['sum'] = te['te_0->'] + te['te_1->'] + te['te_3->'] + te['te_4->']
dyn_te = te.merge(dyn, on='rule')
plt.figure()
print('# with transfer entropy', dyn_te.shape[0])
print('sum transfer entropy v. transient:', spearmanr(dyn_te['sum'], dyn_te['mean_transient']))
plt.scatter(dyn_te['sum'], dyn_te['mean_transient'])
plt.xlabel('Transfer Entropy (bits)')
plt.ylabel('Transient Length (time steps)')
plt.yscale('log')
plt.tight_layout()
plt.savefig('plots/k5/dynamics/te_transient.pdf') 
plt.savefig('plots/k5/dynamics/te_transient.png') 
plt.savefig('plots/k5/dynamics/te_transient.svg') 


# AIS and transient length
te_dyn = ais.merge(dyn, on="rule")
print('AIS v. transient:', spearmanr(te_dyn['ais'], te_dyn['mean_transient']))
plt.figure()
plt.scatter(te_dyn['ais'], te_dyn['mean_transient'])
plt.xlabel('Active Information Storage (bit)')
plt.ylabel('Transient Length (timesteps)')
plt.yscale('log')
plt.tight_layout()
plt.savefig('plots/k5/dynamics/ais_dynamics.pdf')
plt.savefig('plots/k5/dynamics/ais_dynamics.png')
plt.savefig('plots/k5/dynamics/ais_dynamics.svg')


# O-information and transient length
o_dyn = o_info.merge(dyn, on="rule")
print('# entries with O-info:', o_dyn.shape[0])
print('O-information v. transient:', spearmanr(o_dyn['o-info'], o_dyn['mean_transient']))
plt.figure()
plt.scatter(o_dyn['o-info'], o_dyn['mean_transient'])
plt.xlabel('O-information (bit)')
plt.ylabel('Transient Length (timesteps)')
plt.yscale('log')
plt.tight_layout()
plt.savefig('plots/k5/dynamics/o_dynamics.pdf')
plt.savefig('plots/k5/dynamics/o_dynamics.png')
plt.savefig('plots/k5/dynamics/o_dynamics.svg')


print('-'*40)
print('I_pm quantities against transient length')
print('-'*40)


# A bunch of the figures for the PID stuff
ipm_dyn = ipm_sb.merge(dyn_fixed, on='rule').dropna()
print('# of rules with ipm values:', ipm_dyn.shape[0])
fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize=(6,6))

# trasnsient length always on log scale
ax[0,0].set_yscale('log')

# full redundancy
print('pid redundancy:', spearmanr(ipm_dyn['{0}{1}{2}{3}{4}'], ipm_dyn['mean_transient']))
ax[0, 0].scatter(ipm_dyn['{0}{1}{2}{3}{4}'], ipm_dyn['mean_transient'],
                 facecolor='none', edgecolor='k', alpha=0.6, s=10)
ax[0, 0].set_xlabel('Full Redundancy {0}{1}{2}{3}{4} (bit)')
#ax[0, 0].set_ylabel('Transient Length (timesteps)')

# full synergy
print('pid synergy:', spearmanr(ipm_dyn['{0:1:2:3:4}'], ipm_dyn['mean_transient']))
ax[0, 1].scatter(ipm_dyn['{0:1:2:3:4}'], ipm_dyn['mean_transient'],
                 facecolor='none', edgecolor='k', alpha=0.6, s=10)
ax[0, 1].set_xlabel('Full Synergy {0:1:2:3:4} (bit)')
#ax[0, 1].set_ylabel()

# self unique
print('pid self unique:', spearmanr(ipm_dyn['{2}'], ipm_dyn['mean_transient']))
ax[1, 0].scatter(ipm_dyn['{2}'], ipm_dyn['mean_transient'],
                 facecolor='none', edgecolor='k', alpha=0.6, s=10)
ax[1, 0].set_xlabel('Past Self Unique {2} (bit)')


# nonself unique
ipm_dyn['nonself_unique'] = ipm_dyn[['{0}', '{1}', '{3}', '{4}']].mean(axis='columns')
print('pid nonself unique:', spearmanr(ipm_dyn['nonself_unique'], ipm_dyn['mean_transient']))
ax[1, 1].scatter(ipm_dyn['nonself_unique'], ipm_dyn['mean_transient'],
                 facecolor='none', edgecolor='k', alpha=0.6, s=10)
ax[1, 1].set_xlabel('Mean Nonself Unique {0}{1}{3}{4} (bit)')

fig.text(0.015, 0.5, 'Transient Length (timesteps)', ha='center', va='center', rotation='vertical')
plt.tight_layout()
plt.savefig('plots/k5/dynamics/ipm_pid_atoms_transient.pdf')
plt.savefig('plots/k5/dynamics/ipm_pid_atoms_transient.png')
plt.savefig('plots/k5/dynamics/ipm_pid_atoms_transient.svg')


# synergy bias and transient length
fig, ax = plt.subplots()
print('synergy bias:', spearmanr(ipm_dyn['B_syn'], ipm_dyn['mean_transient']))
ax.scatter(ipm_dyn['B_syn'], ipm_dyn['mean_transient'],
                 facecolor='none', edgecolor='k', alpha=0.6, s=10)
ax.set_yscale('log')
ax.set_ylabel('Mean Transient (timestep)')
ax.set_xlabel(r'$B_{syn}$')
plt.tight_layout()
plt.savefig('plots/k5/dynamics/ipm_bsyn_transient.pdf')
plt.savefig('plots/k5/dynamics/ipm_bsyn_transient.png')
plt.savefig('plots/k5/dynamics/ipm_bsyn_transient.svg')


# synergy bias and transient length
ipm_dyn = ipm_ps.merge(dyn_fixed, on = 'rule').dropna()
print('# rules with pure synergy:', ipm_dyn.shape[0])
fig, ax = plt.subplots()
print('pure synergy:', spearmanr(ipm_dyn['pure_synergy'], ipm_dyn['mean_transient']))
ax.scatter(ipm_dyn['pure_synergy'], ipm_dyn['mean_transient'],
                 facecolor='none', edgecolor='k', alpha=0.6, s=10)
ax.set_yscale('log')
ax.set_ylabel('Mean Transient (timestep)')
ax.set_xlabel(r'Pure Synergy')
plt.tight_layout()
plt.savefig('plots/k5/dynamics/ipm_psyn_transient.pdf')
plt.savefig('plots/k5/dynamics/ipm_psyn_transient.png')
plt.savefig('plots/k5/dynamics/ipm_psyn_transient.svg')


print('-'*40)
print('I_min quantities against transient length')
print('-'*40)


# A bunch of the figures for the PID stuff
imin_dyn = imin_sb.merge(dyn_fixed, on='rule').dropna()
print('# of rules with imin values:', imin_dyn.shape[0])
fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize=(6,6))

# trasnsient length always on log scale
ax[0,0].set_yscale('log')

# full redundancy
print('pid redundancy:', spearmanr(imin_dyn['{0}{1}{2}{3}{4}'], imin_dyn['mean_transient']))
ax[0, 0].scatter(imin_dyn['{0}{1}{2}{3}{4}'], imin_dyn['mean_transient'],
                 facecolor='none', edgecolor='k', alpha=0.6, s=10)
ax[0, 0].set_xlabel('Full Redundancy {0}{1}{2}{3}{4} (bit)')
#ax[0, 0].set_ylabel('Transient Length (timesteps)')

# full synergy
print('pid synergy:', spearmanr(imin_dyn['{0:1:2:3:4}'], imin_dyn['mean_transient']))
ax[0, 1].scatter(imin_dyn['{0:1:2:3:4}'], imin_dyn['mean_transient'],
                 facecolor='none', edgecolor='k', alpha=0.6, s=10)
ax[0, 1].set_xlabel('Full Synergy {0:1:2:3:4} (bit)')
#ax[0, 1].set_ylabel()

# self unique
print('pid self unique:', spearmanr(imin_dyn['{2}'], imin_dyn['mean_transient']))
ax[1, 0].scatter(imin_dyn['{2}'], imin_dyn['mean_transient'],
                 facecolor='none', edgecolor='k', alpha=0.6, s=10)
ax[1, 0].set_xlabel('Past Self Unique {2} (bit)')


# nonself unique
imin_dyn['nonself_unique'] = imin_dyn[['{0}', '{1}', '{3}', '{4}']].mean(axis='columns')
print('pid nonself unique:', spearmanr(imin_dyn['nonself_unique'], imin_dyn['mean_transient']))
ax[1, 1].scatter(imin_dyn['nonself_unique'], imin_dyn['mean_transient'],
                 facecolor='none', edgecolor='k', alpha=0.6, s=10)
ax[1, 1].set_xlabel('Mean Nonself Unique {0}{1}{3}{4} (bit)')

fig.text(0.015, 0.5, 'Transient Length (timesteps)', ha='center', va='center', rotation='vertical')
plt.tight_layout()
plt.savefig('plots/k5/dynamics/imin_pid_atoms_transient.pdf')
plt.savefig('plots/k5/dynamics/imin_pid_atoms_transient.png')
plt.savefig('plots/k5/dynamics/imin_pid_atoms_transient.svg')


# synergy bias and transient length
fig, ax = plt.subplots()
print('synergy bias:', spearmanr(imin_dyn['B_syn'], imin_dyn['mean_transient']))
ax.scatter(imin_dyn['B_syn'], imin_dyn['mean_transient'],
                 facecolor='none', edgecolor='k', alpha=0.6, s=10)
ax.set_yscale('log')
ax.set_ylabel('Mean Transient (timestep)')
ax.set_xlabel(r'$B_{syn}$')
plt.tight_layout()
plt.savefig('plots/k5/dynamics/imin_bsyn_transient.pdf')
plt.savefig('plots/k5/dynamics/imin_bsyn_transient.png')
plt.savefig('plots/k5/dynamics/imin_bsyn_transient.svg')


# pure synergy and transient length
imin_dyn = imin_ps.merge(dyn_fixed, on = 'rule').dropna()
print('# rules with pure synergy:', imin_dyn.shape[0])
fig, ax = plt.subplots()
print('pure synergy:', spearmanr(imin_dyn['pure_synergy'], imin_dyn['mean_transient']))
ax.scatter(imin_dyn['pure_synergy'], imin_dyn['mean_transient'],
                 facecolor='none', edgecolor='k', alpha=0.6, s=10)
ax.set_yscale('log')
ax.set_ylabel('Mean Transient (timestep)')
ax.set_xlabel(r'Pure Synergy')
plt.tight_layout()
plt.savefig('plots/k5/dynamics/imin_psyn_transient.pdf')
plt.savefig('plots/k5/dynamics/imin_psyn_transient.png')
plt.savefig('plots/k5/dynamics/imin_psyn_transient.svg')


# switch gears to redundancy stats
print('-'*40)
print('Canalization measures and transient length')
print('-'*40)

# get ourselves a dataframne
ca_dyn = cana.merge(dyn_fixed, on='rule')

# Efective connectivity
fig, ax = plt.subplots()
print('effective connectivity:', spearmanr(ca_dyn['ke'], ca_dyn['mean_transient']))
ax.scatter(ca_dyn['ke'], ca_dyn['mean_transient'],
                 facecolor='none', edgecolor='k', alpha=0.4, s=20)
ax.set_yscale('log')
ax.set_ylabel('Mean Transient (timestep)')
ax.set_xlabel(r'$k_{e}$')
plt.tight_layout()
plt.savefig('plots/k5/dynamics/ke_transient.pdf')
plt.savefig('plots/k5/dynamics/ke_transient.png')
plt.savefig('plots/k5/dynamics/ke_transient.svg')


# Asymmetry
fig, ax = plt.subplots()
print('asymmetry:', spearmanr(ca_dyn['ka'], ca_dyn['mean_transient']))
ax.scatter(ca_dyn['ka'], ca_dyn['mean_transient'],
                 facecolor='none', edgecolor='k', s=20)
ax.set_yscale('log')
ax.set_ylabel('Mean Transient (timestep)')
ax.set_xlabel(r'$k_{a}$')
plt.tight_layout()
plt.savefig('plots/k5/dynamics/ka_transient.pdf')
plt.savefig('plots/k5/dynamics/ka_transient.png')
plt.savefig('plots/k5/dynamics/ka_transient.svg')