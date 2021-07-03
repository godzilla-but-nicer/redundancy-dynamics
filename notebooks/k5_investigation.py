#%% [markdown]
# # Whats going on with $k=5$ CAs
# We're going to start by looking for correlations and stuff between all these
# different measures that we've done Forst we'll do "stuff with transients"
#
# Load the data, its in lots of different files!
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import spearmanr

#%%
# set lambda of interest
lambda_vals = [1,2,3,9,10,11,12,13,14,15,16,17,18,19,22,27,28,29]
# lambda_vals = [1]

# lists that I can use to concatenate my dataframes
dyn_list = []
cana_list = []

for lamb in lambda_vals:
    # load the dynamics data
    dyn = pd.read_csv('../data/k5/attractors/lambda_' + str(lamb) + '_attractors.csv',
                      index_col=0)
    # have to do some really dumb conversion
    dyn['rule'] = dyn['rule'].astype(int)
    
    # lets just average over transients and periods here
    trans = dyn[dyn['measure'] == 'transient'].copy().drop('measure', axis=1)
    trans['transient'] = trans['value']
    trans = trans[['rule', 'trial', 'transient']]
    trans['unq_ind'] = range(0, trans.shape[0])
    print(trans.shape)

    peri = dyn[dyn['measure'] == 'period'].copy().drop('measure', axis=1)
    peri['period'] = peri['value']
    peri = peri[['rule', 'trial', 'period']]
    peri['unq_ind'] = range(0, peri.shape[0])
    print(peri.shape)


    # combine and clean. get mean of feautres by rule
    dyn1 = pd.merge(peri, trans, how='left', on=['unq_ind'])
    dyn1['lambda'] = lamb
    dyn1['rule'] = dyn1['rule_x']
    dyn1['period_transient'] = dyn1['period'] + dyn1['transient']
    dyn1 = dyn1.fillna(64000)
    dyn = (dyn1[['lambda', 'rule', 'period_transient']]
           .groupby('rule')
           .mean()
           .reset_index())


    # load the canalization data
    cana = pd.read_csv('../data/k5/stats/k5_cana_lambda_' + str(lamb) + '.csv', 
                       index_col=0)
    cana['rule'] = dyn['rule'].astype(int)

    # add to the lists
    dyn_list.append(dyn)
    cana_list.append(cana)

# load the o information data frame that is all in one thing
o_info_df = pd.read_csv('../data/k5/stats/o_information.csv', index_col=0)
o_info_df['rule'] = o_info_df['rule'].astype(int)

# load the info dynamics dataframes and combine them
te_df = pd.read_csv('../data/k5/stats/te_rules.csv', index_col=0)
ais_df = pd.read_csv('../data/k5/stats/ais_rules.csv', index_col=0)
info_dyn = pd.merge(te_df, ais_df, on='rule')
info_dyn[info_dyn == -1] = np.NAN

# stack the dataframes
dyn_df = pd.concat(dyn_list)
cana_df = pd.concat(cana_list)
# %% [markdown]
# ## Let's try  and make that langton dynamics plot
#%%
lambda_means = dyn_df.groupby('lambda').mean()
fig, ax = plt.subplots()
ax.plot(lambda_means.index / 30, lambda_means['period_transient'])
ax.set_yscale('log')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('Period + Transient')
plt.savefig('../plots/k5/langton_copy.pdf')
# %% [markdown]
# ## Ok now lets get o-information in the mix
#%%
# combine
o_info_dyn = pd.merge(dyn_df, o_info_df, on='rule')
total = pd.merge(o_info_dyn, cana_df, on='rule')

# lets add a function for our plotting regression lines
def regline(model, x_vals):
    return model.params[1] * x_vals + model.params[0]


# filter out insignificant o-information values
o_info_dyn = o_info_dyn[(o_info_dyn['p'] == 1) | (o_info_dyn['p'] == 0)]

# do the regression
fit_linreg = sm.OLS(np.log(o_info_dyn['period_transient']), 
                    sm.add_constant(o_info_dyn['o-information'])).fit()
reg_lims = np.arange(min(total['o-information']), max(total['o-information']), 0.01)
line_y = regline(fit_linreg, reg_lims)

# lets try a spearman correlation
o_corr = spearmanr(o_info_dyn['o-information'], o_info_dyn['period_transient'])
print(o_corr)

# plot scatter plot, regression line, vertical line at zero
fig, ax = plt.subplots()
ax.axvline(0, color='grey', linestyle='--')
ax.scatter(o_info_dyn['o-information'], o_info_dyn['period_transient'], 
           s=10, alpha=0.5)
#ax.text(0.5, 10**4, r'$\rho={:.3f}$'.format(o_corr.correlation))
ax.set_yscale('log')
ax.set_ylabel('Period + Transient')
ax.set_xlabel('O-information')
plt.tight_layout()
plt.savefig('../plots/k5/o-info.pdf')
plt.show()

# %% [markdown]
# ## ok now let's get the canalization data in the mix
#%%

ke_regress = sm.OLS(np.log(total['period_transient']), sm.add_constant(1-total['kr*'])).fit()
ke_lims = np.arange(min(1-total['kr*']), max(1-total['kr*']), 0.01)

# get spearman correlations
ke_corr = spearmanr((1-total['kr*']), total['period_transient'])
ks_corr = spearmanr(total['ks*'], total['period_transient'])

fig, ax = plt.subplots(sharey=True)
ax.scatter((1 - total['kr*']), total['period_transient'], s=10, alpha=0.5)
#ax.text(0.32, 10**4, r'$\rho={:.3f}$'.format(ke_corr[0]), ha='center')
ax.set_xlabel(r'$k_e$')
ax.set_ylabel('Period + Transient')

# ax[1].scatter(total['ks*'], total['period_transient'], s=10, alpha=0.7)
# ax[1].set_xlabel(r'$k_s^*$')
# ax[1].text(0.82, 10**4, r'$\rho={:.3f}$'.format(ks_corr[0]), ha='center')
ax.set_yscale('log')
plt.savefig('../plots/k5/ke_dynamics.pdf')
plt.tight_layout()

# %% [markdown]
# ## Ok now lets do O-information and canalization!
#%%
o_ke_corr = spearmanr((1-total['kr*']), total['o-information'])
print(o_ke_corr)

oke_reg = sm.OLS(total['o-information'], sm.add_constant(1-total['kr*'])).fit()
oke_lims = np.linspace(min(1-total['kr*']), max(1-total['kr*']), 20)
oke_y = regline(oke_reg, oke_lims)
print(oke_reg.summary())

fig, ax = plt.subplots()
ax.scatter((1-total['kr*']), total['o-information'], s=10, alpha=0.5)
ax.plot(oke_lims, oke_y, color='C1', linestyle='--')
ax.set_xlabel(r'$k_e$')
ax.set_ylabel(r'O-information [bits]')
plt.tight_layout()
plt.savefig('../plots/k5/ke_o_info_reg.pdf')
plt.show()
# %% [markdown]
te_cana = pd.merge(info_dyn, cana_df, on='rule').dropna()
fig, ax = plt.subplots(2, 5, figsize=(8, 3))

# 0
# redundancy
ax[0, 0].scatter(te_cana['0->'], te_cana['r(0)'])
corr = spearmanr(te_cana['0->'], te_cana['r(0)'])
if corr[1] < 0.05:
    ax[0, 0].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')
# symmetry
ax[1, 0].scatter(te_cana['0->'], te_cana['s(0)'])
corr = spearmanr(te_cana['0->'], te_cana['s(0)'])
if corr[1] < 0.05:
    ax[1, 0].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')


# 1
# redundancy
ax[0, 1].scatter(te_cana['1->'], te_cana['r(1)'])
corr = spearmanr(te_cana['1->'], te_cana['r(1)'])
if corr[1] < 0.05:
    ax[0, 1].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')
# symmetry
ax[1, 1].scatter(te_cana['1->'], te_cana['s(1)'])
corr = spearmanr(te_cana['0->'], te_cana['s(0)'])
if corr[1] < 0.05:
    ax[1, 1].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')

# 2
# redundancy
ax[0, 2].scatter(te_cana['ais'], te_cana['r(2)'])
corr = spearmanr(te_cana['ais'], te_cana['r(2)'])
if corr[1] < 0.05:
    ax[0, 2].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')
# symmetry
ax[1, 2].scatter(te_cana['ais'], te_cana['s(2)'])
corr = spearmanr(te_cana['0->'], te_cana['s(0)'])
if corr[1] < 0.05:
    ax[1, 2].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')

# 3
# redundancy
ax[0, 3].scatter(te_cana['3->'], te_cana['r(3)'])
corr = spearmanr(te_cana['3->'], te_cana['r(3)'])
if corr[1] < 0.05:
    ax[0, 3].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')
# symmetry
ax[1, 3].scatter(te_cana['3->'], te_cana['s(3)'])
corr = spearmanr(te_cana['0->'], te_cana['s(0)'])
if corr[1] < 0.05:
    ax[1, 3].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')

# 4
# redundancy
ax[0, 4].scatter(te_cana['4->'], te_cana['r(4)'])
corr = spearmanr(te_cana['4->'], te_cana['r(4)'])
if corr[1] < 0.05:
    ax[0, 4].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')
# symmetry
ax[1, 4].scatter(te_cana['4->'], te_cana['s(4)'])
corr = spearmanr(te_cana['0->'], te_cana['s(0)'])
if corr[1] < 0.05:
    ax[1, 4].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')

ax[1, 0].set_xlabel(r'$T_{0 \rightarrow 2}$')
ax[1, 1].set_xlabel(r'$T_{1 \rightarrow 2}$')
ax[1, 2].set_xlabel(r'$AIS_2$')
ax[1, 3].set_xlabel(r'$T_{3 \rightarrow 2}$')
ax[1, 4].set_xlabel(r'$T_{4 \rightarrow 2}$')
ax[0, 0].set_ylabel(r'$r(i)$')
ax[1, 0].set_ylabel(r'$s(i)$')
plt.tight_layout()
# %%
corr_mat = te_cana.drop(['rule', 's(0)', 's(1)', 's(2)', 's(3)', 's(4)', 'kr*', 'ks*'], axis=1).corr()
sns.heatmap(corr_mat, mask=np.triu(corr_mat), annot=True)
# %%
te_cana = pd.merge(info_dyn, cana_df, on='rule').dropna()
fig, ax = plt.subplots(ncols=5, sharey=True, figsize=(8, 2))

# 0
# redundancy
ax[0].scatter(te_cana['0->'], te_cana['r(0)'], alpha = 0.7, s=10)
corr = spearmanr(te_cana['0->'], te_cana['r(0)'])
if corr[1] < 0.05:
    ax[0, 0].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')


# 1
# redundancy
ax[1].scatter(te_cana['1->'], te_cana['r(1)'], alpha=0.7, s=10)
corr = spearmanr(te_cana['1->'], te_cana['r(1)'])
if corr[1] < 0.05:
    ax[0, 1].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')

# 2
# redundancy
ax[2].scatter(te_cana['ais'], te_cana['r(2)'], alpha= 0.7, s=10)
corr = spearmanr(te_cana['ais'], te_cana['r(2)'])
if corr[1] < 0.05:
    ax[0, 2].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')

# 3
# redundancy
ax[3].scatter(te_cana['3->'], te_cana['r(3)'], alpha=0.7, s=10)
corr = spearmanr(te_cana['3->'], te_cana['r(3)'])
if corr[1] < 0.05:
    ax[0, 3].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')

# 4
# redundancy
ax[4].scatter(te_cana['4->'], te_cana['r(4)'], alpha=0.7, s=10)
corr = spearmanr(te_cana['4->'], te_cana['r(4)'])
if corr[1] < 0.05:
    ax[0, 4].text(0.25, 0.5, r'\rho={:.2f}'.format(corr[0]), ha='center')

ax[0].set_xlabel(r'$T_{0 \rightarrow 2}$')
ax[1].set_xlabel(r'$T_{1 \rightarrow 2}$')
ax[2].set_xlabel(r'$AIS_2$')
ax[3].set_xlabel(r'$T_{3 \rightarrow 2}$')
ax[4].set_xlabel(r'$T_{4 \rightarrow 2}$')
ax[0].set_ylabel(r'$r(i)$')
plt.tight_layout()
plt.savefig('../plots/k5/transfer.pdf')
# %% [markdown]
# ## Synergy Bias in $k=5$ CA
#
# ### Comparing the two methods
# First we'll start with comparing the synergy bias in both $I_{min}$ and 
# $I_{\pm}$. Maybe they are the same!?
# %%
imin_sb = pd.read_csv('../data/k5/stats/imin_synergy_bias.csv')
imin_sb['B_syn_imin'] = imin_sb['B_syn']
ipm_sb = pd.read_csv('../data/k5/stats/ipm_synergy_bias.csv')
ipm_sb['B_syn_ipm'] = ipm_sb['B_syn']

sb = (imin_sb.drop('B_syn', axis = 1)
             .merge(ipm_sb.drop('B_syn', axis = 1), on = 'rule'))

plt.figure(figsize=(6,6))
plt.scatter(sb['B_syn_imin'], sb['B_syn_ipm'], alpha=0.4, s=100)
plt.plot([0,1], [0,1], linestyle = '--', c='k')
plt.xlabel(r'$B_{syn} \;\; [I_{min}]$')
plt.ylabel(r'$B_{syn} \;\; [I_{\pm}]$')
plt.show()
# %% [markdown]
# 
# ### O-information and synergy bias
#
#  Is O-information capturing synergy?
#
# %%
info = sb.merge(o_info_dyn, on = 'rule')
info = info.merge(te_cana, on = 'rule')

print(spearmanr(info['B_syn_imin'], info['o-information']))

linreg = sm.OLS(info['o-information'], sm.add_constant(info['B_syn_imin'])).fit()
reg_lims = np.array([min(info['B_syn_imin']), max(info['B_syn_imin'])])
print(linreg.summary())


plt.figure()

plt.scatter(info['B_syn_imin'], info['o-information'], alpha = 0.6)

plt.plot(reg_lims, linreg.params[1] * reg_lims + linreg.params[0],
         linestyle='--', color='C1')

plt.xlabel(r'$B_{syn} \;\; [I_{min}]$')
plt.ylabel(r'O-information')
plt.show()
# %% [markdown]
# It kind of is!? Correlated at least. 
# 
# ### Effective connectivity and synergy bias
# %%
info = total.merge(sb, on = 'rule')
print(spearmanr(info['B_syn_imin'], 1 - info['kr*']))

linreg = sm.OLS(1 - info['kr*'], sm.add_constant(info['B_syn_imin'])).fit()
reg_lims = np.array([min(info['B_syn_imin']), max(info['B_syn_imin'])])
print(linreg.summary())

ke_vals = 1 - info['kr*']
imin_sb = info['B_syn_ipm']
# %%
plt.figure(figsize=(4, 4))
g = sns.JointGrid(x=ke_vals, y=imin_sb)
g.plot(sns.scatterplot, sns.histplot)
plt.xlabel(r'$k_e$', fontsize=14)
plt.ylabel(r'$B_{syn} \;\; [I_{min}]$', fontsize=14)
plt.tight_layout()
plt.savefig('../plots/k5/imin_ke.png', pad_inches=0.2)
plt.show()
# %%
# ### Maybe synergy bias and transient length will be cool
#
# %%

plt.figure(figsize=(4, 4))
g = sns.JointGrid(x=info['B_syn_ipm'], y=np.log(info['period_transient']))
g.plot(sns.scatterplot, sns.histplot)
plt.xlabel(r'$B_{syn} \;\; [I_{min}]$', fontsize=14)
plt.ylabel(r'$Period + Transient$', fontsize=14)
plt.tight_layout()
plt.savefig('../plots/k5/imin_transient.png', pad_inches=0.2)
plt.show()
# %%
