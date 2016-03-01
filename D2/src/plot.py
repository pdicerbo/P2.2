import numpy as np
import matplotlib.pyplot as plt

# min_files = ["results/explicit_res.dat", "results/min_check_conj.dat"]
# notice that the main.x produces files with extension .dat, but on github I store only ".safe" copy of this files
min_files = ["results/explicit_res.safe", "results/min_check_conj.safe"]

plt.figure()

for nfile in min_files:
    data = np.loadtxt(nfile)
    n_iter = data[:,0]
    Fx = data[:,1]
    if nfile == min_files[0]:
        R_inn = data[:,2]
        plt.loglog(n_iter, Fx, label = "$|r^{expl}|$")
        plt.loglog(n_iter, R_inn, label = "$|r^{inn}|$")
        plt.title("Explicit Error Check\nObtained fixing Matrix Size = 150, CondNumb = $10^5$ and $\hat{r} = 10^{-3}$")
        plt.xlabel("N_Iter")
        plt.ylabel("$|r|$")
        # plt.show()
        plt.savefig("results/err_check.png")
        plt.close("all")
        
    elif nfile == min_files[1]:
        plt.semilogx(n_iter, Fx)
        plt.title("Minimization Check\nObtained fixing Matrix Size = 150, CondNumb = $10^5$ and $\hat{r} = 10^{-3}$")
        plt.xlabel("N_Iter")
        plt.ylabel("F(x)")
        # plt.show()
        plt.savefig("results/min_check.png")
        plt.close("all")
