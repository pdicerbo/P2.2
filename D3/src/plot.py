import numpy as np
import matplotlib.pyplot as plt

nfiles = ["results/strong_timing.dat", "results/weak_timing.dat"]
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

        plt.errorbar(x_real, y_real[0]/y_real, yerr=err, label = namef[:-4])
        plt.errorbar(x_real, x_real, yerr=err, label = namef[:-4])
        plt.title('Strong Scaling\nfor MPI version with $\hat{r}_{targ} = 10^{-15}$ and Matrix Size = 120000')
        plt.xlabel('NPE')
        plt.ylabel('Speedup')
        plt.savefig("results/first_scaling.png")
        plt.close('all')

    elif namef == nfiles[1]:

        plt.errorbar(x_real, y_real, yerr=err, label = namef[:-4])
        plt.title('Scaling obtained with $\hat{r}_{targ} = 10^{-15}$ Matrix Size = 12000 * NPE')
        plt.xlabel('Matrix size')
        plt.ylabel('time (s)')
        plt.savefig("results/second_scaling.png")
        plt.close('all')
