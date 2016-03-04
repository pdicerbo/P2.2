import numpy as np
import matplotlib.pyplot as plt

# nfiles = ["results/strong_timing.safe", "results/strong_timing.vec", "results/weak_timing.dat", "results/weak_timing.vec"]
nfiles = ["results/strong_timing.safe", "results/strong_timing.vec", "results/weak_timing.safe", "results/weak_timing.vec"]
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
      
    if namef == nfiles[0]:
        nproc = np.copy(x_real)
        strong = np.copy(y_real)
        s_err = np.copy(err)

    if namef == nfiles[1]:
        strong_v = np.copy(y_real)
        sv_err = np.copy(err)

    if namef == nfiles[2]:
        sizem = np.copy(x_real)
        weak = np.copy(y_real)
        w_err = np.copy(err)

    if namef == nfiles[3]:
        weak_v = np.copy(y_real)
        wv_err = np.copy(err)


plt.errorbar(nproc, strong[0]/strong, yerr=s_err, label = "normal")
plt.errorbar(nproc, strong_v[0]/strong_v, yerr=sv_err, label = "vect")
plt.errorbar(nproc, nproc, yerr=err, label = "linear")
plt.title('Strong Scaling\nfor MPI version with $\hat{r}_{targ} = 10^{-15}$ and Matrix Size = 5$\cdot 10^5$')
plt.xlabel('NPE')
plt.ylabel('Speedup')
plt.legend()
plt.savefig("results/first_scaling.png")
plt.close('all')

plt.errorbar(nproc, strong, yerr=s_err, label = "normal")
plt.errorbar(nproc, strong_v, yerr=sv_err, label = "vect")
plt.title('Execution time\nfor MPI version with $\hat{r}_{targ} = 10^{-15}$ and Matrix Size = 5$\cdot 10^5$')
plt.xlabel('NPE')
plt.ylabel('time (s)')
plt.legend()
plt.savefig("results/exec_time.png")
plt.close('all')

plt.errorbar(sizem, weak, yerr=w_err, label = "normal")
plt.errorbar(sizem, weak_v, yerr=wv_err, label = "vect")
plt.title('Scaling obtained with $\hat{r}_{targ} = 10^{-15}$ Matrix Size = 12000 * NPE')
plt.xlabel('Matrix Size')
plt.ylabel('time (s)')
plt.legend()
plt.savefig("results/second_scaling.png")
plt.close('all')
