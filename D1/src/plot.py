import numpy as np
import matplotlib.pyplot as plt

# nfiles = ["results/first_scaling.dat", "results/sec_scaling.dat"]
# notice that the main.x produces files with extension .dat, but on github I store only ".safe" copy of this files
nfiles = ["results/first_scaling.safe", "results/sec_scaling.safe"]
rep = 10     # number of repetition per measure, set in top of src/main.c

for namef in nfiles:
    data = np.loadtxt(namef)
    X = data[:,0]
    Y   = data[:,1]

    n_elem = len(data) / rep

    x_real = np.zeros(n_elem)
    y_real = np.zeros(n_elem)
    err = np.zeros(n_elem)

    i = 0
    j = 0
    count  = 0
    y_tmp1 = 0.

    while i < n_elem:
        while j < len(X):
            # performing mean value calculation
            y_tmp1 += Y[i + j]
            j += n_elem


        x_real[count] = X[i]
        y_real[count] = y_tmp1 / rep

        y_tmp1 = 0.
        j = 0 # need to perform error calculation
        
        while j < len(X):
            # performing error calculation
            y_tmp1 += (Y[i + j] - y_real[count])**2
            j += n_elem

        err[count] = (y_tmp1 / (rep - 1.))**0.5

        y_tmp1 = 0.
        count += 1
        i += 1
        j = 0
        
    plt.figure()

    if namef == nfiles[0]:

        plt.errorbar(x_real, y_real, yerr=err, label = namef[:-4])
        plt.title('Scaling obtained with $\hat{r}_{targ} = 10^{-10}$ and $CondNumb = 10^6$')
        plt.xlabel('Matrix Size')
        plt.ylabel('N_Iter')
        plt.savefig("results/first_scaling.png")
        plt.close('all')

    elif namef == nfiles[1]:

        plt.errorbar(x_real**0.5, y_real, yerr=err, label = namef[:-4])
        plt.title('Scaling obtained with $\hat{r}_{targ} = 10^{-10}$ Matrix Size = 500')
        plt.xlabel('sqrt(Condition Number)')
        plt.ylabel('N_Iter')
        plt.savefig("results/second_scaling.png")
        plt.close('all')

# MINIMIZATION CHECK PLOT

min_files = ["results/min_check_grad.dat", "results/min_check_conj.dat"]

plt.figure()

for nfile in min_files:
    data = np.loadtxt(nfile)
    n_iter = data[:,0]
    Fx = data[:,1]
    plt.semilogx(n_iter, Fx, label = nfile[18:-4])
    
plt.title("Minimization Check\nObtained fixing Matrix Size = 150, CondNumb = $10^5$ and $\hat{r} = 10^{-3}$")
plt.xlabel("N_Iter")
plt.ylabel("F(x)")
plt.legend()
# plt.show()
plt.savefig("results/min_check.png")
