import os
import pandas as pd

lambda_vals = range(7, 25)

submit_file = '/N/u/patgwall/BigRed3/redundancy-dynamics/slurmy.script'

out_str = """#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=patgwall@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-{0}:00:00
#SBATCH --partition=general
#SBATCH --job-name=ca_lambda_{1}
#SBATCH --output=logs/ca_lambda_{1}.log

######  Module commands #####



######  Job commands go below this line #####
cd ~/redundancy-dynamics/
python scripts/CA1D/run_k5_lambda.py 64 {1} 100 100"""


for lamb in lambda_vals:
    submit_str = out_str.format(24, lamb)
    with open(submit_file, 'w') as fout:
        fout.write(submit_str)
    os.system('sbatch ' + submit_file)
