import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sns.set_theme(style='darkgrid')

N = sys.argv[1]
pid = sys.argv[2]

# load dataframe
dyn = pd.read_csv('data/eca/attractors/eca_' + N + '_summary.csv', index_col=None)
info = pd.read_csv('data/eca/' + pid + '_df.csv')
info_noisy = info.drop('rule', axis=1) + np.random.uniform(-0.01, 0.01, size=(256, 19))
info_noisy['rule'] = info['rule']
df = dyn.merge(info_noisy, on='rule')

# calculated columns
df['log_mean_tra'] = np.log(df['mean_transient']+1)
df['log_var_tra'] = np.log(np.sqrt(df['var_transient']+1))
df['coef_variation'] = np.sqrt(df['var_transient']) / df['mean_transient']
df['excess_synergy'] = df['((0, 1, 2),)'] - df['((0,), (1,), (2,))']

# mean transient and shared information
fig, ax = plt.subplots()
sns.scatterplot(x='((0,), (1,), (2,))', y='log_mean_tra', hue='class',
                style='class', s=60, ax=ax, data=df, palette='Set1')
ax.set_xlabel(r'Full Shared Information [bit]')
ax.set_ylabel(r'log(Mean Transient)')
plt.tight_layout()
plt.savefig('plots/eca/mean_transient_by_shared_' + pid + '.png')

# mean transient and synergy
fig, ax = plt.subplots()
sns.scatterplot(x='((0, 1, 2),)', y='log_mean_tra', hue='class',
                style='class', s=60, ax=ax, data=df, palette='Set1')
ax.set_xlabel(r'Full Synergistic Information [bit]')
ax.set_ylabel(r'log(Mean Transient)')
plt.tight_layout()
plt.savefig('plots/eca/mean_transient_by_synergy_' + pid + '.png')

# mean_transient and excess synergy
fig, ax = plt.subplots()
sns.scatterplot(x='excess_synergy', y='log_mean_tra', hue='class',
                style='class', s=60, ax=ax, data=df, palette='Set1')
ax.set_xlabel(r'Excess Synergy [bit]')
ax.set_ylabel(r'log(Mean Transient)')
plt.tight_layout()
plt.savefig('plots/eca/mean_transient_by_excess_synergy_' + pid + '.png')

# unique side
fig, ax = plt.subplots()
sns.scatterplot(x='((0,),)', y='log_mean_tra', hue='class',
                style='class', s=60, ax=ax, data=df, palette='Set1')
ax.set_xlabel(r'Side Unique Information [bit]')
ax.set_ylabel(r'log(Mean Transient)')
plt.tight_layout()
plt.savefig('plots/eca/mean_transient_by_unq_side_' + pid + '.png')

# unique center
fig, ax = plt.subplots()
sns.scatterplot(x='((1,),)', y='log_mean_tra', hue='class',
                style='class', s=60, ax=ax, data=df, palette='Set1')
ax.set_xlabel(r'Center Unique Information [bit]')
ax.set_ylabel(r'log(Mean Transient)')
plt.tight_layout()
plt.savefig('plots/eca/mean_transient_by_unq_center_' + pid + '.png')
