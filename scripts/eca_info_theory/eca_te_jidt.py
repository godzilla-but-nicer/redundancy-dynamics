import numpy as np
import pandas as pd
import sys
from glob import glob
from casim.CA1D import CA1D
from jpype import *


# for testing
files = glob('data/eca/runs/*')
rules = [int(f.split('/')[-1].split('_')[0]) for f in files]

row_list = []

# Add JIDT jar library to the path
jarLocation = "/home/patgwall/projects/jidt/infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# 1. Construct the calculator:
calcClass = JPackage(
    "infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
calc = calcClass(2, 12, 1, 1, 1, 1)
# 3. Initialise the calculator for (re-)use:



for rule in rules:
    in_file_name = 'data/eca/runs/' + str(rule) + '_500_300_100.csv'
    data = np.loadtxt(in_file_name, delimiter=',').astype(int)
    te_vals = np.zeros(4) - 1
    row = {}
    for ip, pair in enumerate([(0, 1), (2, 1)]):
        source = JArray(JInt,1)(data[:100000, pair[0]].tolist())
        destination = JArray(JInt, 1)(data[:100000, pair[1]].tolist())

        calc.initialise()
        # 4. Supply the sample data:
        calc.addObservations(source, destination)
        # 5. Compute the estimate:
        result = calc.computeAverageLocalOfObservations()
        # 6. Compute the (statistical significance via) null distribution empirically (e.g. with 100 permutations):
        measDist = calc.computeSignificance(100)

        print("TE_Kraskov (KSG)(col_%d -> col_%d) = %.4f nats (null: %.4f +/- %.4f std dev.; p(surrogate > measured)=%.5f from %d surrogates)" %
                (pair[0], pair[1], result, measDist.getMeanOfDistribution(), measDist.getStdOfDistribution(), measDist.pValue, 100))

        if measDist.pValue < 0.05:
            te_vals[ip] = result

    row['rule'] = rule
    row['0->'] = te_vals[0]
    row['2->'] = te_vals[1]
    row_list.append(row)

df = pd.DataFrame(row_list)
df.to_csv('data/eca/stats/te_rules.csv')
