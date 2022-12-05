import pandas as pd
import numpy as np

n_prods = 8
n_crits = 12

crits = []
crits_norm = []

for i in range(n_crits):
    crits.append(np.zeros(n_prods))
    crits_norm.append(np.zeros(n_prods))

data = pd.read_excel('data.xls').values

for i in range(n_crits):
    for t in range(1, 9):
        if (data[i][9] == 'макс'):
            crits[i][t-1] = 1 / data[i][t]
            continue
        crits[i][t-1] = data[i][t]

weights = np.ones(n_crits)
weights_norm = np.zeros(n_crits)
for i in range(n_crits):
    weights_norm[i] = weights[i] / sum(weights)

def voronin():
    effic = np.zeros(n_prods)
    sums = np.zeros(n_crits)

    for i in range(n_crits):
        for t in range(n_prods):
            sums[i] = sums[i] + crits[i][t]

    for i in range(n_crits):
        for t in range(n_prods):
            crits_norm[i][t] = crits[i][t] / sums[i]

    for i in range(n_prods):
        for t in range(n_crits):
            effic[i] = ( effic[i] +
               weights_norm[i] * (1 - crits_norm[t][i]) ** (-1)
            )

    return effic

def main():
    res = voronin()

    min = 1000
    index = None
    for i in range(n_prods):
        if min > res[i]:
            min = res[i]
            index = i

    print('Integrated performance evaluation:')
    for i in range(n_prods):
        if i == index:
            print('prod', i+1, '-', res[i], '- best')
            continue
        print('prod', i+1, '-', res[i])


main()