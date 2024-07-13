import CTOT
from matplotlib import pyplot as plt
import numpy as np

# Set up to store results of color transfer experiment with each algorithm
timeHis = []
operHis = []
algos = ['sinkhorn', 'greenkhorn', 'stochSinkhorn', 'sag', 'apdagd', 'aam']
for algo in algos:
    times = []
    opers = []
    A = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    for a in A:
        print(f"Running {algo} with acc={1/a}")
        t, o = CTOT.colTrans(1/a, algo)
        times.append(t)
        opers.append(o)
    timeHis.append(times)
    operHis.append(opers)

# Plotting runtimes and number of arithmetic operations
plt.figure(5, figsize=(8,4))
for i, algo in enumerate(algos):
    plt.plot(A, timeHis[i], label=algo)
plt.legend(loc='upper left', ncol=7, bbox_to_anchor=(-0.1, 1.15))
plt.xlabel('1/eps')
plt.ylabel('required times')
plt.savefig(f"results/times")
plt.figure(6, figsize=(8,4))
for i, algo in enumerate(algos):
    plt.plot(A, np.log10(operHis[i]), label=algo)
plt.legend(loc='upper left', ncol=7, bbox_to_anchor=(-0.1, 1.15))
plt.xlabel('1/eps')
plt.ylabel('log_10(# of operations)')
plt.savefig(f"results/opers")