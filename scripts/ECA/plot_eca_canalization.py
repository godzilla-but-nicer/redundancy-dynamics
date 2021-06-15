import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sns.set_theme(style='darkgrid')


N = int(sys.argv[1])

dyn = pd.read_csv('data/eca/attractors/eca_' + str(N) + '_summary.csv', index_col=None)
cana = pd.read_csv('data/eca/canalization_df.csv', index_col=None)
cana_noisy = cana.drop('rule', axis=1) + np.random.uniform(-0.01, 0.01, size=cana.drop('rule', axis=1).shape)
cana_noisy['rule'] = cana['rule']
df = dyn.merge(cana_noisy, on='rule')

# we need to calculate some features
df['ke'] = 1 - df['kr*']
df['log_mean_tra'] = np.log(df['mean_transient']+1)
df['log_var_tra'] = np.log(np.sqrt(df['var_transient']+1))
df['coef_variation'] = np.sqrt(df['var_transient']) / df['mean_transient']

# mean vs ke
fig, ax = plt.subplots()
sns.scatterplot(x='ke', y='log_mean_tra', hue='class', style='class', data=df, 
                ax=ax, palette='Set1', s=60)
ax.set_xlabel(r'$k_e$')
ax.set_ylabel(r'$\ln$(Transient)')
plt.tight_layout()
plt.savefig('plots/eca/mean_transient_by_ke.png')

# coef of var on ke
fig, ax = plt.subplots()
sns.scatterplot(x='ke', y='coef_variation', hue='class', style='class', data=df, 
                ax=ax, palette='Set1', s=60)
ax.set_xlabel(r'$k_e$')
ax.set_ylabel(r'$c_v$')
plt.tight_layout()
plt.savefig('plots/eca/coef_transient_by_ke.png')

fig, ax = plt.subplots()
sns.scatterplot(x='ks*', y='log_mean_tra', hue='class', style='class', data=df, 
                ax=ax, palette='Set1', s=60, size='log_var_tra')
# ax.scatter(df['ks*'], np.log(df['var_transient']+1), label='var')
ax.set_xlabel(r'$ks$')
ax.set_ylabel(r'$\ln$(Transient)')
plt.savefig('plots/eca/mean_transient_by_ks.png')
