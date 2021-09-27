# %% [markdown]
# # CA Analysis $k=5$
# This is my second attempt to look at the CA with $k=5$. this time i've
# sampled a bit more evenly and hopefully its all good baby
# %%
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

rules = pd.read_csv('../data/k5/sampled_rules.csv', index_col=0)
ipm_full = pd.read_csv('../data/k5/stats/ipm_synergy_bias.csv', index_col=0).reset_index()
#ipm = ipm_full.merge(rules, on='rule').dropna()
ipm = ipm_full.dropna()
ipm['rule'] = ipm['rule'].astype(int)
# %% [markdown]
# ## Synergy Bias Distribution
# Last time our samples were highly skewed toward high synergy bias. Is this
# Still true? Our sampling still isnt perfect.
# %%
print('# Samples with valid B_syn: ', ipm.shape[0])
sns.histplot(ipm['B_syn'])
plt.xlabel('Synergy Bias')
plt.savefig('../plots/k5/bsyn_hist.png')
plt.savefig('../plots/k5/bsyn_hist.pdf')
plt.show()
# %% [markdown]
# Still skewed but thats ok maybe hopefully
# ## Effective Connectivity
# what we really want to know about is how effective connectivity compares
# to synergy bias. In ECA we get a strong relationship but it vanished with the
# $k=5$ with the older sampling.
#
# For the sake of exploration we'll start with a distribution of effective
# connectivities. I think I expect this to have a peak somewhere in the upper
# half of the range
# %%
cana = pd.read_csv('../data/k5/stats/k5_cana.csv', index_col=0)
ipm_cana = ipm.merge(cana, on='rule')
ipm_cana['ke*'] = 1 - ipm_cana['kr*']
sns.histplot(ipm_cana['ke*'], kde=True)
plt.savefig('../plots/k5/ke_hist.pdf')
plt.savefig('../plots/k5/ke_hist.png')

# %% [markdown]
# ## $k_e^*$ and $B_{syn}$
# This comparison is really why we're here

# %%

print(spearmanr(ipm_cana['ke*'], ipm_cana['B_syn']))

sns.scatterplot(x='B_syn', y='ke*', hue='mutual_info', data=ipm_cana)
plt.savefig('../plots/k5/ke_vs_bsyn.png')
plt.savefig('../plots/k5/ke_vs_bsyn.pdf')
plt.show()
#%% [markdown]
# Its a weird shape and the mutual information doesn't really seem to show
# a pattern in in terms of where in this space it deviates from 1. Let's take a
# look at the distribution before we move on to get a better sense of what is
# going on with it.

# ## MI distribution
#
# I might need to know how mutual information is distributed
# so lets take a look.

# %%
sns.histplot(ipm['mutual_info'])
plt.xlabel(r'$I({l_2^{t-1}, l_1^{t-1}, c^{t-1}, r_1^{t-1}, r_2^{t-1}}:c^t)$')
plt.savefig('../plots/k5/mi_hist.pdf')
plt.savefig('../plots/k5/mi_hist.png')
plt.show()
# %% [markdown]
# 
# thats not super helpful although its pretty clear that 'deviates from 1' ia
# the right way to think about it. I'm not sure if that makes sense or not.
# I would think that we should either always have a full bit or rarely have a
# a full bit given that this is a deterministic but often chaotic system and im
# estimating probabilities for the joint states. Maybe thats just it, my
# estimates aren't that good and I should ignore MI??
#
# ## Regression
#
# Ok so  correlates (spearman's r) can we do regression? It looks like OLS
# might just work?
# %% [markdown]

# set up weighted least squares linear regression
X = sm.add_constant(ipm_cana['B_syn'])
y = ipm_cana['ke*']
linreg = sm.WLS(y, X, weights=ipm_cana['mutual_info']).fit()
olsfit = sm.OLS(y, X).fit()

# lets get the residuals
ipm_cana['pred_ols'] = linreg.params[0] + linreg.params[1] * ipm_cana['ke*']
ipm_cana['resi_ols'] = ipm_cana['B_syn'] - ipm_cana['pred_ols']

# plot the distribution of residuals and the residuals themselves
fig = plt.figure(constrained_layout=True, figsize=(8, 8))
ax = fig.subplot_mosaic([['A', 'B'],
                         ['C', 'C']])

# residuals themselves on left
ax['A'].scatter(ipm_cana['ke*'], ipm_cana['resi_ols'], facecolor='none', 
                edgecolor='grey')
ax['A'].axhline(0, color='C1', linestyle='dotted', linewidth = 5)
ax['A'].set_xlabel(r'$k_e^*$')
ax['A'].set_ylabel('obs - pred')

# distribution
sns.histplot(ipm_cana['resi_ols'], ax=ax['B'])
ax['B'].set_xlabel(r'obs - pred')
ax['B'].set_ylabel('Count')

# the fit itself
# the data
ax['C'].scatter(ipm_cana['B_syn'], ipm_cana['ke*'], facecolor='none', 
                edgecolor='C0', label='data')

# the WLS fit
wls_ll = round(linreg.llf, 2)
ols_ll = round(olsfit.llf, 2)

ax['C'].plot(ipm_cana['B_syn'], linreg.fittedvalues, 'g--',
             label=r'WLS $l={}$'.format(wls_ll))
ax['C'].plot(ipm_cana['B_syn'], olsfit.fittedvalues, '--', color='C1', 
             label=r'OLS $l={}$'.format(ols_ll))


# labels
ax['C'].legend()
# save it
plt.savefig('../plots/k5/bsyn_ke_reg.pdf')
plt.savefig('../plots/k5/bsyn_ke_reg.png')

# %% [markdown]
# # O-information
#
# Now let's take a look at O-information to see if it reports on effective
# connectivity. We will also take a look at how well it correlates with 
# redundancy in the form of $1 - B_{syn}$
# %%
o_info = pd.read_csv('../data/k5/stats/o_information_new.csv', index_col=0)
ipm = ipm_cana.merge(o_info, on='rule')

# drop unsignif. values. this needs to have a multiple testing correction prob.
# for bonferoni, my p values dont have enough resolution.
sig_o_info = ipm[(ipm['p'] > 0.95) | (ipm['p'] < 0.05)][['rule', 'B_syn', 'ke*', 'o-info']]

# make the plot for the comparison with synergy bias
sig_o_info['B_red'] = 1 - sig_o_info['B_syn']
fig, ax = plt.subplots()
sns.scatterplot(x=sig_o_info['B_red'], y=sig_o_info['o-info'], ax=ax)
ax.set_xlabel(r'$1 - B_{syn}$')
ax.set_ylabel('O-information')
plt.savefig('../plots/k5/bsyn_oinfo.pdf')
plt.savefig('../plots/k5/bsyn_oinfo.png')
plt.show()

# lets get a spearman correlation too
print(spearmanr(sig_o_info['B_red'], sig_o_info['o-info']))
# %% [markdown]
# not the most impressive relationship.
#
# ## O-info and $k_e^*$
#
# the more important one anyway.
# %%
sns.scatterplot(sig_o_info['ke*'], sig_o_info['o-info'])
plt.savefig('../plots/k5/ke_oinfo.pdf')
plt.savefig('../plots/k5/ke_oinfo.png')
plt.show()
print(spearmanr(sig_o_info['ke*'], sig_o_info['o-info']))

# %% [markdown]
# Uncorrelated! thats weird. it doesn't really seem like O-information is
# as useful as we might like.
#
# # Directed Information Measures
#
# Transfer entropy and active information storage tell us about when
# the past of a variable is useful for the prediction of another variable. This
# really should not work for all variables in highly canalized functions with
# low effective connectivity.
# %%
directed = pd.read_csv('../data/k5/stats/directed.csv', index_col=0)
ipm_dir = ipm_cana.merge(directed, on='rule').replace(-1, np.nan)

# let's get all of the like 'same input' transfer entropy vs. redundancy pairs
input_specific = ['rule', 'r(0)', 'r(1)', 'r(2)', 'r(3)', 'r(4)', 
                          '0->', '1->', 'ais', '3->', '4->']
rename_cols = {'r(0)': 'cana_0',
               'r(1)': 'cana_1',
               'r(2)': 'cana_2',
               'r(3)': 'cana_3',
               'r(4)': 'cana_4',
               '0->' : 'info_0',
               '1->' : 'info_1',
               'ais' : 'info_2',
               '3->' : 'info_3',
               '4->' : 'info_4',}
dir_info = ipm_dir[input_specific].rename(rename_cols, axis=1).dropna()
directed_long = pd.wide_to_long(dir_info, ['cana', 'info'], 'rule', 'input', sep='_')

# do the plot
plt.figure()
(sns.jointplot(x='info', y='cana', data=directed_long, kind='hist')
    .set_axis_labels(r'$T_{i \rightarrow c}$ // $AIS_c$', r'$r^*(i)$'))
plt.savefig('../plots/k5/directed_cana.pdf')
plt.savefig('../plots/k5/directed_cana.png')
plt.show()
print(spearmanr(directed_long['info'], directed_long['cana']))
# %% [markdown]
#
# So that seems weird it implies that there must be a bunch of redundant
# information complicating these relationships.
#
# ## Directed info and $B_{syn}$
#
# Can we see evidence for this influential redundant informaiton as a negative
# correlation between $B_{syn}$ and a sum of these measures
# %%
ipm_dir['info_sum'] = (ipm_dir['0->'] + ipm_dir['1->'] 
                       + ipm_dir['3->'] + ipm_dir['4->'])

ipm_dir = ipm_dir.dropna()

plt.figure()
plt.scatter(ipm_dir['B_syn'], ipm_dir['info_sum'])
plt.xlabel(r'$B_{syn}$')
plt.ylabel(r'$\sum T_{i \rightarrow c}$')
plt.savefig('../plots/k5/tesum_bsyn.pdf')
plt.savefig('../plots/k5/tesum_bsyn.png')
plt.show()
print(spearmanr(ipm_dir['info_sum'], ipm_dir['B_syn']))
# %% [markdown]
# Slight negative correlation. This makes sense. I think rather than rely on
# this relationship we are probably more interested in the TE conditioned on
# all other variables.
#
# # Lambda
#
# I sampled rule tables using langton's lamdba which in a binary system is very
# similar to output entropy. Are any of the patterns simply products of lambda?
# 
# ## Correlation as a function of lambda
# 
# first we will look at the correlation between effective connectivity as a
# function of lambda
# %%
# This shit doesn't work
# from scipy.stats import entropy
# ls = []
# corrs = []
# ipm_cana['entropy'] = entropy([(ipm_cana['lambda'] + 2) / 32, 1 - (ipm_cana['lambda'] + 2) / 32])
# for l in ipm_cana['lambda'].unique():
#     ls.append(l)
#     ldf = ipm_cana[ipm_cana['lambda'] == l]
#     rp = spearmanr(ldf['B_syn'], ldf['ke*'])
#     if rp.pvalue < 0.05:
#         corrs.append(rp.correlation)
#     else:
#         corrs.append(0)
# plt.scatter(ls, corrs)
# plt.xlabel(r'$\lambda$')
# plt.ylabel(r'Spearman $\rho$')
# plt.savefig('../plots/k5/lambda_corr.pdf')
# plt.savefig('../plots/k5/lambda_corr.png')
# plt.figure()
# 
# plt.scatter(ipm_cana['entropy'], ipm_cana['B_syn'])
# plt.xlabel(r'$H_{out}$')
# plt.ylabel(r'$B_{syn}$')
# plt.savefig('../plots/k5/out_ent_bsyn.pdf')
# plt.savefig('../plots/k5/out_ent_bsyn.png')
# plt.show()
# %% [markdown]
#
# Dynamics
#
# Might have to rerun dynamics calculations for the ones I care about here but who knows.
# I think I will. Anyway we basically care about whether any of these measures
# tell us anything about the dynamics.
# %%
raw_dyn = pd.read_csv('../data/k5/combined_dynamics.csv', index_col=0)
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

ipm_dyn = ipm_cana.merge(dyn, on='rule')

# %% [markdown]
#
# ## Distribution of transients

# %%
sns.histplot(dyn['mean_transient'], log_scale=True, bins=20)
plt.savefig('../plots/k5/transient_hist.pdf')
plt.savefig('../plots/k5/transient_hist.png')
plt.show()

# %% [markdown]
#
# ## dynamics and b_syn

# %%
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(ipm_dyn['B_syn'], ipm_dyn['mean_transient'], 
           facecolors='none', edgecolors='C0')
ax.set_yscale('log')
ax.set_xlabel(r'$B_{syn}$')
ax.set_ylabel(r'Transient')
plt.ylim((.1, 10**4))
plt.tight_layout()
plt.savefig('../plots/k5/bsyn_dyn.pdf')
plt.savefig('../plots/k5/bsyn_dyn.svg')
plt.savefig('../plots/k5/bsyn_dyn.png')
plt.show()

print(spearmanr(ipm_dyn['B_syn'], ipm_dyn['mean_transient']))
# %% [markdown]
#
# ## dynamics and ke

# %%
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(ipm_dyn['ke*'], ipm_dyn['mean_transient'],
           facecolors='none', edgecolors='C0')
ax.set_yscale('log')
ax.set_xlabel(r'$k_e^*$')
ax.set_ylabel(r'Transient')
plt.ylim((.1, 10**5))
plt.tight_layout()
plt.savefig('../plots/k5/ke_dyn.pdf')
plt.savefig('../plots/k5/ke_dyn.svg')
plt.savefig('../plots/k5/ke_dyn.png')
plt.show()
print(spearmanr(ipm_dyn['ke*'], ipm_dyn['mean_transient']))
# %% [markdown]
#
# ## dynamics and output entropy

# %%
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


dyn['entropy'] = dyn['rule'].apply(lambda x: rule_to_ent(x))

ent_vals = sorted(np.unique(dyn['entropy']))

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

print(len(periods), len(ent_vals), len(se_periods))

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
plt.savefig('../plots/k5/entropy_dynamics.pdf')
plt.savefig('../plots/k5/entropy_dynamics.svg')
plt.savefig('../plots/k5/entropy_dynamics.png')
plt.show()
# %% [markdown]
# ## relationships between ke and b syn and system dynamics
# %%
print(spearmanr(ipm_cana['ke*'], ipm_cana['B_syn']))
ipm_dyn['log_mean_transient'] = np.log(ipm_dyn['mean_transient'])

sns.scatterplot(x='B_syn', y='ke*', hue='log_mean_transient', 
                data=ipm_dyn, palette='cividis', alpha=0.7)
plt.text(0.13, 0.6, r'$\rho={:.3f}$'.format(spearmanr(ipm_cana['ke*'], ipm_cana['B_syn'])[0]))
plt.ylabel(r'$k_e^*$')
plt.xlabel(r'$B_{syn}$')
plt.legend(title=r'$ln(l)$')
plt.savefig('../plots/k5/ke_vs_bsyn_dyn.png')
plt.savefig('../plots/k5/ke_vs_bsyn_dyn.pdf')
plt.show()
# %% [markdown]
# # All dynamics in one plot
# lets get the dynamics plots all in one place for $k=5$

# %%
fig = plt.figure(constrained_layout=True, figsize=(8, 6))
ax = fig.subplot_mosaic([['A', 'A'],
                         ['B', 'C']])

# Transient vs ke
rho_Bsyn = spearmanr(ipm_dyn['B_syn'], ipm_dyn['mean_transient'])[0]
ax['B'].scatter(ipm_dyn['B_syn'], ipm_dyn['mean_transient'],
           facecolors='none', edgecolors='C0')
ax['B'].text(0.1, 10**2.5, r'$\rho = {:.3f}$'.format(rho_Bsyn))
ax['B'].set_yscale('log')
ax['B'].set_xlim(-0.05, 1.05)
ax['B'].set_xlabel(r'$B_{syn}$')
ax['B'].set_ylabel(r'Transient')

rho_ke = spearmanr(ipm_dyn['ke*'], ipm_dyn['mean_transient'])[0]
ax['C'].scatter(ipm_dyn['ke*'], ipm_dyn['mean_transient'],
           facecolors='none', edgecolors='C0')
ax['C'].text(0.1, 10**2.5, r'$\rho = {:.3f}$'.format(rho_ke))
ax['C'].set_xlim(-0.05, 1.05)
ax['C'].set_yscale('log')
ax['C'].set_xlabel(r'$k_e^*$')

# ipm_dyn['entropy'] = ipm_dyn['rule'].apply(lambda x: rule_to_ent(x))

# ent_vals = sorted(np.unique(ipm_dyn['entropy']))

# se_periods = []
# periods = []
# se_transients = []
# transients = []
# for l in ent_vals:
    # ld = ipm_dyn[ipm_dyn['entropy'] == l]
    # periods.append(np.mean(ld['mean_period'].dropna()))
    # se_periods.append(np.std(ld['mean_period'].dropna() / np.sqrt(len(ld['mean_period']))))
    # transients.append(np.mean(ld['mean_transient'].dropna()))
    # se_transients.append(np.std(ld['mean_transient'].dropna() / np.sqrt(len(ld['mean_transient']))))
# print(len(ent_vals))

# # convert all to numpy arrays for easy math later
# se_periods = np.array(se_periods)
# periods = np.array(periods)
# se_transients = np.array(se_transients)
# transients = np.array(transients)

ax['A'].plot(ent_vals, periods, label='Period', marker='^', mfc='white', mec='C0')
ax['A'].fill_between(ent_vals, periods - se_periods, periods + se_periods, color='C0', alpha = 0.4)
ax['A'].plot(ent_vals, transients, label='Transient',
         marker='s', mfc='white', mec='C1')
ax['A'].fill_between(ent_vals, transients - se_transients, transients + se_transients, color='C1', alpha = 0.4)
ax['A'].set_xlabel(r'$H_{out}$')
ax['A'].set_ylabel(r'Timesteps')
ax['A'].set_yscale('log')
ax['A'].legend(loc='upper left')

# get things situated
plt.tight_layout()
plt.savefig('../plots/k5/all_dynamics.pdf')
plt.savefig('../plots/k5/all_dynamics.svg')
plt.savefig('../plots/k5/all_dynamics.png')
plt.show()

# %%
# Looking at specific PI atoms


## combining all of the raw ipm data into opne dataframe

# Before we can look at specific values we need the data containing those values.
# Right now each rule lives in its own file. We need to combine them into one file.
# We'll do that below and also save a big dataframe.
#
# While we're at it we're going to beautify some of the columns of this thing
#%%
from glob import glob
import re

def pretty_labels_map(atom_labels):
    """
    transforms all of these crazy tuples into the notation used in williams and
    beer I_min PID
    """
    rename_map = {}
    for label in atom_labels:
        new_label = str(label)

        # eliminate commas and spaces
        new_label = new_label.replace(',', '')
        new_label = new_label.replace(' ', '')

        # replace braces
        new_label = new_label.replace('(', '{')
        new_label = new_label.replace(')', '}')

        # separate digits with colons
        while re.search(r'\d\d', new_label):
            new_label = re.sub(r'(\d)(\d)', r'\1:\2', new_label)

        # put them in a map
        rename_map[label] = new_label[1:-1]

    return rename_map

# indiv_ipm = glob('../data/k5/pid/raw_ipm/*_ipm.csv')
# df_list = []
# for fin in indiv_ipm:
#     df_list.append(pd.read_csv(fin, index_col=0))
# 
# big_df = pd.concat(df_list)
# lab_map = pretty_labels_map(big_df.columns)
# lab_map['rule'] = 'rule'
# 
# big_df = big_df.rename(columns=lab_map)
# big_df.to_csv('../data/k5/pid/ipm.csv')
big_df = pd.read_csv('../data/k5/pid/ipm.csv', index_col=0)
# %%
# ## Keep only interpretable columns
#
# Lots of these columns are really hard to think about. we dont care about those.
# I'll also go ahead and add the canalization data here. Finally I'll also add
# synergy bias so we can see
#%%
keep_cols = ['rule', '{0:1:2:3:4}', '{0}{1}{2}{3}{4}', 
             '{0}', '{1}','{2}','{3}','{4}']
ipm_singles = big_df[keep_cols]
ipm_singles = ipm_singles.merge(cana, on='rule')
ipm_singles = ipm_singles.merge(ipm_full, on = 'rule').drop(['mutual_info'], axis=1)
# %% [markdown]
#
# ## Time to get these correlations
#
# We're going to do a correlation matrix for all of these things. Really we
# just want to see the information measures on the x-axis and the canalization
# on the y-axis.
# %%
# First of all we need to select only our sample
ipm_singles_rules = ipm_singles.merge(rules.drop('lambda', axis=1), on = 'rule')
# think im going to drop all of the symmetry values too
# ipm_singles = ipm_singles.drop(['ks*', 's(0)', 's(1)', 's(2)', 's(3)', 's(4)'], axis=1)

# I want effective connectivity instead of kr
ipm_singles_rules['ke'] = 1 - ipm_singles_rules['kr*']

# add asterisk to names of columns with nans
df_marked = ipm_singles_rules.drop(['rule', 'kr*'], axis=1)

# next we want to reorder the columns
col_list = list(df_marked.columns)
col_list.insert(7, 'B_syn')
col_list.insert(8, 'ke')
df_marked = df_marked[col_list[:-2]]

# and we want to relabel them
label_map = {'B_syn': r'$B_{syn}$',
             'ke': r'$k_e$',
             'r(0)': r'$r(0)$',
             'r(1)': r'$r(1)$',
             'r(2)': r'$r(2)$',
             'r(3)': r'$r(3)$',
             'r(4)': r'$r(4)$'}

df_marked = df_marked.rename(columns=label_map)

# calculate the correlations
df_corr = df_marked.corr(method='spearman')

print(df_corr)


# get the correlations with pandas
mat = sns.heatmap(df_corr, annot=False, xticklabels=df_marked.columns,
                    yticklabels=df_marked.columns,
                    cbar_kws={'label': 'Correlation'},
                    cmap='RdBu', center=0)
plt.axvline(x=8, c='k', alpha=0.7, ls=':')
plt.axhline(y=8, c='k', alpha=0.7, ls=':')
plt.savefig('../plots/k5/corr_mat.png')
plt.savefig('../plots/k5/corr_mat.pdf')


# %% [markdown]
# %% [markdown]
# # Imin and Ipm
# %%
imin_df = pd.read_csv('../data/k5/pid/imin.csv', index_col=0)
imin_sb = pd.read_csv('../data/k5/stats/imin_synergy_bias.csv', index_col=0)
imin_singles = imin_df[keep_cols]
imin_singles = imin_singles.merge(imin_sb, on = 'rule')

# need to make sure we have the same rows in both dataframes
imin_rules = imin_singles[['rule']]
ipm_imin_corr = ipm_singles.merge(imin_rules, on='rule').dropna()
intercept_rules = ipm_imin_corr[['rule']]
imin_ipm_corr = imin_singles.merge(intercept_rules, on='rule').dropna()

labels = imin_singles.columns[1:]
corrs = []
for lab in labels:
    corrs.append(spearmanr(imin_ipm_corr[lab], ipm_imin_corr[lab])[0])

plt.figure()
plt.bar(labels, corrs)
plt.xticks(rotation=50)
plt.ylabel(r'$\rho(I_{min},\; I_\pm)$')
plt.savefig('plots/k5/imin_ipm_compare.png')
plt.savefig('plots/k5/imin_ipm_compare.pdf')
# %%
plt.figure()
plt.scatter(imin_ipm_corr['B_syn'], ipm_imin_corr['B_syn'])
# %%
