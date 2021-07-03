# %% [markdown]
# # Sample the $k=5$ rules
# We need to get a subset of k=5 rules to run PID on that isnt just me picking. 
# We have a big list of rules sorted by lambda so we can use that. Let's pull
# out like 30 from each lambda value or something and make a new list of rules
# for PID, other information theoretic analysis. We dont have all lambda values
# but thats probably ok. We can sample in $\lambda \in [10,19]$.
# %%
import pandas as pd
import numpy as np
from casim.CA1D import CA1D
rng = np.random.default_rng()

info_rules = []
lambdas = []
l_range = range(1, 30)
for l in l_range:
    rules = (pd.read_csv(
             '../data/k5/attractors/lambda_' + str(l) + '_attractors.csv', 
             index_col=0)
             ['rule']
             .unique()
             .astype(int))
    samples = rng.choice(rules, size = 20)
    ls = [l]*20
    info_rules.extend(samples)
    lambdas.extend(ls)

# we dont have dynamics for the some of the high lambda rules but we want them
# anyway. we can get the dynamics later
ca = CA1D(5, 0, random_seed=1234)
for _ in range(20):
    info_rules.append(ca.lambda_rule(30))
    lambdas.append(30)

df_rules = pd.DataFrame({'lambda': lambdas, 'rule': info_rules})
df_rules.to_csv('../data/k5/sampled_rules.csv')
# %%
