#%% [markdown]
# ## ECA information theory comparison figures and stuff
#%% [markdown]
# ## Load packages and data
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cana_df = pd.read_csv("../data/eca/canalization_df.csv")
imin_df = pd.read_csv("../data/eca/imin_df.csv", index_col = 0)
ipm_df = pd.read_csv("../data/eca/pm_df.csv", index_col = 0)
unq_rules = pd.read_csv("../data/eca/eca_equiv_classes.csv")
#%% [markdown]
## Write a function for Thomas' synergy bias
#%%
# have to hard code a bunch of shit because i dont have lattice information
def synergy_bias(df_row):
    # make a list of lists to look up each level
    l1 = ['((0,), (1,), (2,))']
    l2 = ["((0,), (1,))", "((0,), (2,))", "((1,), (2,))"]
    l3 = ["((0,), (1, 2))","((1,), (0, 2))","((2,), (0, 1))"]
    l4 = ["((0,),)","((1,),)","((2,),)","((0, 1), (0, 2), (1, 2))"]
    l5 = ["((0, 1), (0, 2))","((0, 1), (1, 2))","((0, 2), (1, 2))"]
    l6 = ["((0, 1),)","((0, 2),)","((1, 2),)"]
    l7 = ["((0, 1, 2),)"]
    l_atoms = [l1, l2, l3, l4, l5, l6, l7]
    
    # now we can calculate the thing
    bias = 0
    total_pi = df_row.drop('rule', axis = 1).sum(axis=1).values[0]
    print(total_pi)
    for i in range(7):
        l_sum = 0
        for atom in l_atoms[i]:
            l_sum += df_row[atom].values[0]
        bias += (i + 1) / 7 * l_sum / total_pi
    
    return bias

#%% [markdown]
## Synergy Bias against effective connectivity

#%%
rules = unq_rules['rule'].unique()
imin_sb = []
ipm_sb = []
ke_vals = []
for rule in rules:
    cana_row = cana_df[cana_df['rule'] == rule]
    imin_row = imin_df[imin_df['rule'] == rule]
    ipm_row = ipm_df[ipm_df['rule'] == rule]

    imin_sb.append(synergy_bias(imin_row))
    ipm_sb.append(synergy_bias(ipm_row))
    ke_vals.append(1 - cana_row['kr*'].values[0])
# %%
plt.figure(figsize=(4,4))
g = sns.JointGrid(x=ke_vals, y=imin_sb)
g.plot(sns.regplot, sns.histplot)
plt.xlabel(r'$k_e$', fontsize=14)
plt.ylabel(r'$B_{syn} (I_{min})$', fontsize=14)
plt.tight_layout()
plt.savefig('../plots/eca/imin_ke.png', pad_inches=0.2)
plt.show()
# %%
plt.figure(figsize=(4,4))
g = sns.JointGrid(x=ke_vals, y=ipm_sb)
g.plot(sns.regplot, sns.histplot)
plt.xlabel(r'$k_e$', fontsize=14)
plt.ylabel(r'$B_{syn}(I_{\pm})$', fontsize=14)
plt.tight_layout()
plt.savefig('../plots/eca/ipm_ke.png', pad_inches=0.2)
plt.show()
# %%
