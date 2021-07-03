import sys
import numpy as np
from casim.CA1D import CA1D

# unpack command line args
rule = int(sys.argv[1])
N = int(sys.argv[2])
max_steps = int(sys.argv[3])
runs = int(sys.argv[4])

# we're going to do a shit ton of cells for not that many steps and then 
# save only 5 columns for our TE calculations just to make it play nice with JIDT
ca = CA1D(5, rule, random_seed = 1234)
unq_starts = []
for i in range(runs):
    ca.set_state(ca.rng.choice(2, size=N))
    time_series = ca.simulate_time_series(N, max_steps)
    print(time_series.shape)
    unq_starts.append(time_series[50:,:])

# combine all of the individual runs into one big one
combined_series = np.vstack(unq_starts)
print(combined_series[:10])
# pull out all possible nonoverlapping sequences of 5
# nonoverlapping shouldnt matter but we just dont need as much data as taking 
# overlapping gives us
neighborhoods = []
for i in range(int(N / 5)):
    neighborhoods.append(combined_series[:, (i*5):(i*5)+5])

neighborhoods = np.vstack(neighborhoods)
print(neighborhoods[:10])
file_name = 'data/k5/runs/' + str(rule) + '_' + str(N) + '_' + str(max_steps) + '_' + str(runs) + '.csv'
np.savetxt(file_name, neighborhoods, delimiter=',', fmt='%d')
