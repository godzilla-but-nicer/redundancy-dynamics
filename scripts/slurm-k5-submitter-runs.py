from glob import glob
import os
import pandas as pd
import sys

files = glob('data/k5/runs/*.csv')
rules = [f.split('/')[-1].split('_')[0] for f in files]
print(rules)
submit_file = '/N/u/patgwall/BigRed3/redundancy-dynamics/slurmy.script'

out_str = """#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=patgwall@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-0:20:00
#SBATCH --partition=general
#SBATCH --job-name=ca_rule_{0}
#SBATCH --output=logs/ca_rule_{0}.log

######  Module commands #####



######  Job commands go below this line #####
cd ~/redundancy-dynamics/
python scripts/CA1D/run_transient_k5.py {0} 500 300 100"""


for rule in rules:
    submit_str = out_str.format(rule)
    with open(submit_file, 'w') as fout:
        fout.write(submit_str)
    os.system('sbatch ' + submit_file)
