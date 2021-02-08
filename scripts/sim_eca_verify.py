import sys
from tqdm import tqdm
from casim.eca import eca_sim

rule = int(sys.argv[1])
N = 16
steps = 200
trials = 1000

period = []
transients = []

for t in tqdm(range(trials)):
    eca = eca_sim(rule, random_seed=1234)
    
    # this is awkward to have to set None
    per, tra = eca.find_approx_attractor(N, steps, None)
    period.append(per)
    transients.append(tra)

data_path = 'data/eca/attractors/rule_' + str(rule) + '/'
with open(data_path + 'approx_16_periods_' + str(rule) + '.txt', 'w') as fout:
    for per in period:
        fout.write(str(per) + '\n')

with open(data_path + 'approx_16_trans_' + str(rule) + '.txt', 'w') as fout:
    for tra in transients:
        fout.write(str(tra) + '\n')