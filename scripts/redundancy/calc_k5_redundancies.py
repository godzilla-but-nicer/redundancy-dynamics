import pandas as pd
import sys
from casim.utils import to_binary
from tqdm import tqdm
from cana.boolean_node import BooleanNode

# rules will come from the dynamics dataframes
l = sys.argv[1]
df_dict = []
df_in = pd.read_csv('data/k5/attractors/lambda_' + str(l) + '_attractors.csv')
rules = df_in['rule'].unique().astype(int)
for rule in rules:
    canal = {}  # becomes row of dataframe

    # to_binary returns list of strings of binary digits
    arr = to_binary(rule, digits=32).astype(str)

    # use CANA to compute canalization
    bn = BooleanNode.from_output_list(outputs=arr, name=rule)
    ks = bn.input_symmetry()
    kr = bn.input_redundancy()
    sym0, sym1, sym2, sym3, sym4 = bn.edge_symmetry()
    red0, red1, red2, red3, red4 = bn.edge_redundancy()

    # update the dictionary with the PI values
    canal['rule'] = rule
    canal['kr*'] = kr
    canal['ks*'] = ks
    canal['r(0)'] = red0
    canal['r(1)'] = red1
    canal['r(2)'] = red2
    canal['r(3)'] = red3
    canal['r(4)'] = red4
    canal['s(0)'] = sym0
    canal['s(1)'] = sym1
    canal['s(2)'] = sym2
    canal['s(3)'] = sym3
    canal['s(4)'] = sym4

    df_dict.append(canal)

# write out the dataframe
df = pd.DataFrame(df_dict)
df_fout = open('data/k5/stats/k5_cana_lambda_' + str(l) + '.csv', 'w')
df.to_csv(df_fout)
df_fout.close()
