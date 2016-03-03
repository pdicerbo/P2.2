import numpy as np
import matplotlib.pyplot as plt

min_files = ["results/explicit_res.dat", "results/min_check_conj.dat"]
# notice that the main.x produces files with extension .dat, but on github I store only ".safe" copy of this files
# min_files = ["results/explicit_res.safe", "results/min_check_conj.safe"]

plt.figure()

for nfile in min_files:
    data = np.loadtxt(nfile)
    n_iter = data[:,0]
    Fx = data[:,1]
    if nfile == min_files[0]:
        R_inn = data[:,2]
        plt.loglog(n_iter, Fx, label = "explicit")
        plt.loglog(n_iter, R_inn, label = "implicit")
        plt.title("Explicit Error Check\nObtained fixing Matrix Size = 150, CondNumb = $10^6$ and $\hat{r} = 10^{-28}$")
        plt.xlabel("N_Iter")
        plt.ylabel("$|r|/|b|$")
        plt.legend()
        plt.savefig("results/err_check.png")
        plt.close("all")
        
    elif nfile == min_files[1]:
        plt.semilogx(n_iter, Fx)
        plt.title("Minimization Check\nObtained fixing Matrix Size = 150, CondNumb = $10^6$ and $\hat{r} = 10^{-28}$")
        plt.xlabel("N_Iter")
        plt.ylabel("F(x)")
        plt.savefig("results/min_check.png")
        plt.close("all")

# laplace_files = ["results/classic_timing.dat", "results/sparse_timing.dat"]
laplace_files = ["results/classic_timing.safe", "results/sparse_timing.safe"]

for namef in laplace_files:
    data = np.loadtxt(namef)
    size_m = data[:,0]
    time = data[:,1]
    plt.plot(size_m, time, label = namef[8:-11])

plt.title("execution time for 1000 repetition")
plt.xlabel("Matrix Size")
plt.ylabel("time (s)")
plt.legend(bbox_to_anchor = (0.26,1.))
plt.savefig("results/timing.png")
plt.close("all")

data = np.loadtxt("results/cond_numb_check.dat")
sig  = data[:,0]
n_it = data[:,1]
plt.plot(sig, n_it)
plt.title("Condition Number Check\nMatrix size = 20000, $\hat{r} = 10^{-8}, \sigma \in [10^{-5}, 0.05]$")
plt.xlabel("(ConditionNumber)**0.5")
plt.ylabel("Number of iteration")
plt.show()

