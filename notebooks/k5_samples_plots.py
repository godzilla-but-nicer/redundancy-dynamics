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
ipm_full = pd.read_csv('../data/k5/stats/ipm_synergy_bias.csv', index_col=0)
#ipm = ipm_full.merge(rules, on='rule').dropna()
ipm = ipm_full.dropna()
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
sig_o_info = ipm[(ipm['p'] > 0.95) | (ipm['p'] < 0.05)][['rule', 'B_syn', 'ke*', 'o-info', 'lambda']]

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
from scipy.stats import entropy
ls = []
corrs = []
ipm_cana['entropy'] = entropy([(ipm_cana['lambda'] + 2) / 32, 1 - (ipm_cana['lambda'] + 2) / 32])
for l in ipm_cana['lambda'].unique():
    ls.append(l)
    ldf = ipm_cana[ipm_cana['lambda'] == l]
    rp = spearmanr(ldf['B_syn'], ldf['ke*'])
    if rp.pvalue < 0.05:
        corrs.append(rp.correlation)
    else:
        corrs.append(0)
plt.scatter(ls, corrs)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'Spearman $\rho$')
plt.savefig('../plots/k5/lambda_corr.pdf')
plt.savefig('../plots/k5/lambda_corr.png')
plt.figure()

plt.scatter(ipm_cana['entropy'], ipm_cana['B_syn'])
plt.xlabel(r'$H_{out}$')
plt.ylabel(r'$B_{syn}$')
plt.savefig('../plots/k5/out_ent_bsyn.pdf')
plt.savefig('../plots/k5/out_ent_bsyn.png')
plt.show()
# %% [markdown]
#
# Dynamics
#
# Might have to rerun dynamics calculations for the ones I care about here but who knows.
# I think I will. Anyway we basically care about whether any of these measures
# tell us anything about the dynamics.
# %%
dyn = pd.read_csv('../data/k5/stats/dynamics.csv', index_col=0)

ipm_dyn = ipm_cana.merge(dyn, on='rule')

# %% [markdown]
#
# ## Distribution of transients

# %%
sns.histplot(ipm_dyn['period_transient'], log_scale=True, bins=20)
plt.savefig('../plots/k5/transient_hist.pdf')
plt.savefig('../plots/k5/transient_hist.png')
plt.show()

# %% [markdown]
#
# ## dynamics and b_syn

# %%
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(ipm_dyn['B_syn'], ipm_dyn['period_transient'], 
           facecolors='none', edgecolors='C0')
ax.set_yscale('log')
ax.set_xlabel(r'$B_{syn}$')
ax.set_ylabel(r'Timesteps')
plt.ylim((1, 10**5))
plt.tight_layout()
plt.savefig('../plots/k5/bsyn_dyn.pdf')
plt.savefig('../plots/k5/bsyn_dyn.svg')
plt.savefig('../plots/k5/bsyn_dyn.png')
plt.show()

print(spearmanr(ipm_dyn['B_syn'], ipm_dyn['period_transient']))
# %% [markdown]
#
# ## dynamics and ke

# %%
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(ipm_dyn['ke*'], ipm_dyn['period_transient'],
           facecolors='none', edgecolors='C0')
ax.set_yscale('log')
ax.set_xlabel(r'$k_e^*$')
ax.set_ylabel(r'Period + Transient')
plt.ylim((1, 10**5))
plt.tight_layout()
plt.savefig('../plots/k5/ke_dyn.pdf')
plt.savefig('../plots/k5/ke_dyn.svg')
plt.savefig('../plots/k5/ke_dyn.png')
plt.show()
print(spearmanr(ipm_dyn['ke*'], ipm_dyn['period_transient']))
# %% [markdown]
#
# ## dynamics and output entropy

# %%
unq_ls = sorted(dyn['lambda'].unique())
zero_prob = np.array([(ul + 1) / 32 for ul in unq_ls])
entropies = [entropy([p, 1-p], base=2) for p in zero_prob]
ent_map = {l: round(e, 4) for l, e in zip(unq_ls, entropies)}
dyn['entropy'] = dyn['lambda'].map(ent_map)

ent_vals = sorted(np.unique(dyn['entropy']))

dynamics = []
periods = []
transients = []
for l in ent_vals:
    ld = dyn[dyn['entropy'] == l]
    dynamics.append(np.mean(ld['period_transient']))
    periods.append(np.mean(ld['period'].dropna()))
    transients.append(np.mean(ld['transient'].dropna()))

plt.figure(figsize=(4,4))
plt.plot(ent_vals, dynamics, label='Period + Transient', marker='o', mfc='white', mec='C0')
plt.plot(ent_vals, periods, label='Period', marker='^', mfc='white', mec='C1')
plt.plot(ent_vals, transients, label='Transient', marker='s', mfc='white', mec='C2')
plt.xlabel(r'$H_{out}$')
plt.ylabel(r'Timesteps')
plt.ylim((1, 10**5))
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
ipm_dyn['log_period_transient'] = np.log(ipm_dyn['period_transient'])

sns.scatterplot(x='B_syn', y='ke*', hue='log_period_transient', 
                data=ipm_dyn, palette='Blues', alpha=0.6)
plt.ylabel(r'$k_e^*$')
plt.xlabel(r'$B_{syn}$')
plt.legend(title=r'$ln(T+l)$')
plt.savefig('../plots/k5/ke_vs_bsyn_dyn.png')
plt.savefig('../plots/k5/ke_vs_bsyn_dyn.pdf')
plt.show()
# %% [markdown]
# # All dynamics in one plot
# lets get the dynamics plots all in one place for $k=5$

# %%
fig, ax = plt.subplots(ncols=3, figsize=(8, 2.66), sharey=True)
ax[1].scatter(ipm_dyn['B_syn'], ipm_dyn['period_transient'],
           facecolors='none', edgecolors='C0')
ax[1].set_yscale('log')
ax[1].set_xlabel(r'$B_{syn}$')
ax[1].set_ylabel(r'Period + Transient')

ax[2].scatter(ipm_dyn['ke*'], ipm_dyn['period_transient'],
           facecolors='none', edgecolors='C0')
ax[2].set_yscale('log')
ax[2].set_xlabel(r'$k_e^*$')
ax[2].set_ylabel(r'Period + Transient')

unq_ls = sorted(dyn['lambda'].unique())
zero_prob = np.array([(ul + 1) / 32 for ul in unq_ls])
entropies = [entropy([p, 1-p], base=2) for p in zero_prob]
ent_map = {l: round(e, 4) for l, e in zip(unq_ls, entropies)}
dyn['entropy'] = dyn['lambda'].map(ent_map)

ent_vals = sorted(np.unique(dyn['entropy']))

dynamics = []
periods = []
transients = []
for l in ent_vals:
    ld = dyn[dyn['entropy'] == l]
    dynamics.append(np.mean(ld['period_transient']))
    periods.append(np.mean(ld['period'].dropna()))
    transients.append(np.mean(ld['transient'].dropna()))

ax[0].plot(ent_vals, dynamics, label='Period + Transient',
         marker='o', mfc='white', mec='C0')
ax[0].plot(ent_vals, periods, label='Period', marker='^', mfc='white', mec='C1')
ax[0].plot(ent_vals, transients, label='Transient',
         marker='s', mfc='white', mec='C2')
ax[0].set_xlabel(r'$H_{out}$')
ax[0].set_ylabel(r'Timesteps')
ax[0].legend(loc='upper left')

# get things situated
plt.tight_layout()
plt.savefig('../plots/k5/all_dynamics.pdf')
plt.savefig('../plots/k5/all_dynamics.svg')
plt.savefig('../plots/k5/all_dynamics.png')
plt.show()

# %%
