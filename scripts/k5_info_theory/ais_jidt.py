import numpy as np
import pandas as pd
import sys
from glob import glob
from casim.CA1D import CA1D
from jpype import *


#for testing
files = glob('data/k5/runs/*')
rules = [int(f.split('/')[-1].split('_')[0]) for f in files]

row_list = []

# Add JIDT jar library to the path
jarLocation = "/home/patgwall/projects/jidt/infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# 1. Construct the calculator:
calcClass = JPackage(
    "infodynamics.measures.discrete").ActiveInformationCalculatorDiscrete
calc = calcClass(2, 12)
# 3. Initialise the calculator for (re-)use:

for rule in rules:
    try: # sometimes this shit just doesnt work
        calc.initialise()
        in_file_name = glob('data/k5/runs/' + str(rule) + '*.csv')[0]
        data = np.loadtxt(in_file_name, delimiter=',').astype(int)
        te_vals = np.zeros(4) - 1
        row = {}
        destination = JArray(JInt, 1)(data[:, 2].tolist())

        # 4. Supply the sample data:
        calc.addObservations(destination)
        # 5. Compute the estimate:
        result = calc.computeAverageLocalOfObservations()
        # 6. Compute the (statistical significance via) null distribution empirically (e.g. with 100 permutations):
        measDist = calc.computeSignificance(100)

        print(rule)
        print("AIS_Discrete(col_2) = %.4f bits (null: %.4f +/- %.4f std dev.; p(surrogate > measured)=%.5f from %d surrogates)" %
              (result, measDist.getMeanOfDistribution(), measDist.getStdOfDistribution(), measDist.pValue, 100))

        if measDist.pValue < 0.05:
            keep_val = result
        else:
            keep_val = -1

        row['rule'] = rule
        row['ais'] = keep_val
        row_list.append(row)
    except:
        continue

df = pd.DataFrame(row_list)
df.to_csv('data/k5/stats/ais_rules.csv')
