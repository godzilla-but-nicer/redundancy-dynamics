import os
import pandas as pd

eca = pd.read_csv('data/eca/eca_equiv_classes.csv', index_col=None)
rules = eca[(eca['rule'] == 106) | (eca['rule'] == 45) | (eca['rule'] == 30)]['rule']
submit_file = '/N/u/patgwall/BigRed3/redundancy-dynamics/slurmy.script'

out_str = """#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=patgwall@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-4:00:00
#SBATCH --partition=general
#SBATCH --mail-type=FAIL
#SBATCH --job-name=eca_{0:d}
#SBATCH --output=logs/eca_{1}.log

######  Module commands #####



######  Job commands go below this line #####
cd ~/redundancy-dynamics/
python scripts/approx_eca_sim.py {2:d} 100 100"""


for rule in rules:
    submit_str = out_str.format(rule, rule, rule)
    with open(submit_file, 'w') as fout:
        fout.write(submit_str)
    os.system('sbatch ' + submit_file)
