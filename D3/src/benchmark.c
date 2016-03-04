#include <stdio.h>
#include <stdlib.h>
#include "../include/inverse_laplace_operator.hpp"
#include "../include/random_gen.hpp"
#include "../include/system_solvers.h"
#include "../include/laplace_utils.h"

#ifdef __MPI
#include <mpi.h>
#endif /* __MPI */

int main(int argc, char** argv){

  double *f, *b, *check_sol;
  double r_hat = 1.e-15;
  double sigma = 0.6, s = -0.5;
  int i, j, n_it, L;
  
  L = 500000; // "full" vector size

#ifdef __MPI

  /* Initialization of the variables needed in the parallelized */
  /* version; each process works on a vector of size L / NumberProcessingElements */
  /* (unless there is a rest, that requires a work redistribution) */
  int NPE, MyID, MyTag, vsize, rest, l_tmp;
  double *results_recv, *b_send;
  int *displ, *recv;
  double tstart, tend;
  FILE* timing;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &MyID);
  MPI_Comm_size(MPI_COMM_WORLD, &NPE);

  /* FOR WEAK SCALING MEASURE */
  /* L *= NPE; */

  MyTag = 42;
  vsize = L;
  L /= NPE;
  rest = vsize % NPE;

  if(rest != 0 && MyID < rest)
    L++;

  /* "initialization" section */
  if(MyID == 0){

    f = (double*) malloc(L * sizeof(double));
    b = (double*) malloc(L * sizeof(double));

    /* process 0 generates the random b vector and sends */
    /* various pieces to the other processes */
    /* while I send data, I initialize also "displs" and "recv" array */
    /* needed to gather the results */
    displ  = (int*) malloc(NPE * sizeof(int));
    recv   = (int*) malloc(NPE * sizeof(int));
    b_send = (double*) malloc(vsize * sizeof(double));

    recv[0] = L;
    displ[0] = 0;

    /* process 0 generates the random b vector and sends */
    /* various pieces to the other processes */
    /* while I send data, I initialize also "displs" and "recv" array */
    /* needed to gather the results */
    fill_source(b, 2.2, 0.5, L);

    /* need use l_tmp because if rest != 0, process 0 have to send bunch */
    /* of array of different size */			      
    l_tmp = L;
    
    for(j = 1; j < NPE; j++){
      
      if(rest != 0 && j == rest)
	l_tmp--;
      
      fill_source(b_send, 2.2, 0.5, l_tmp);
      MPI_Send(b_send, l_tmp, MPI_DOUBLE, j, MyTag, MPI_COMM_WORLD);

      recv[j] = l_tmp;
      displ[j] = displ[j-1]+recv[j-1]; 
    }
    /* initialization of results_recv pointer need from process 0 to gather the results. */
    results_recv = (double*) malloc(vsize * sizeof(double));
  }
  else
    {

    f = (double*) malloc(L * sizeof(double));
    b = (double*) malloc(L * sizeof(double));
    /* processes with MyID != 0 receives in b */
    MPI_Recv(b, L, MPI_DOUBLE, 0, MyTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  }

  /* end of the "initialization" section */
  /* Finally, we performs the calculation and checks the results */  

  tstart = MPI_Wtime();
  
  sparse_conj_grad_alg(f, b, sigma, s, r_hat, L, &n_it, MyID, NPE);

  tend = MPI_Wtime();

  /* process 0 gather the results and checks the correctness */
  MPI_Gatherv(f, L, MPI_DOUBLE, results_recv, recv, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(b, L, MPI_DOUBLE, b_send, recv, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  if(MyID == 0){
    /* STRONG SCALING DATA */
  timing = fopen("results/strong_timing.dat", "a");
  fprintf(timing, "%d\t%lg\n", NPE, tend - tstart);

    /* WEAK SCALING */
  /* timing = fopen("results/weak_timing.dat", "a"); */
  /* fprintf(timing, "%d\t%lg\n", vsize, tend - tstart); */

  fclose(timing);

  if(i == 0)
    printf("\n\tThe found solution is correct\n");
  else
    printf("\n\tThe found solution is wrong\n");
  
  }
#else /* serial version of the code */

  f = (double*) malloc(L * sizeof(double));
  b = (double*) malloc(L * sizeof(double));
  
  /* randomly filling b vector */
  fill_source(b, 2.2, 0.5, L);

  sparse_conj_grad_alg(f, b, sigma, s, r_hat, L, &n_it);

  printf("\n\tSOLUTION\n");
  
  for(j = 0; j < L; j++)
    printf("\t%lg\n", f[j]);

  check_sol = (double*) malloc(L * sizeof(double));

  inverse_laplace_operator(check_sol, b, sigma, L, L);

  printf("\n\tSparse solution:  Check_sol:\n");
  i = 0;

  for(j = 0; j < L; j++){
    printf("\t%lg\t\t  %lg\n", f[j], check_sol[j]);
    if(abs(f[j] - check_sol[j]) > 1.e-14)
      i = 1;
  }

  if(i == 0)
    printf("\n\tThe found solution is correct\n");
  else
    printf("\n\tThe found solution is wrong\n");
 
  free(check_sol);
 
#endif /* __MPI */

  /* going to conclusion... */
  free(f);
  free(b);

#ifdef __MPI

  if(MyID == 0){
    free(results_recv);
    free(displ);
    free(recv);
    free(b_send);

  }

  MPI_Finalize();
  
#endif /* __MPI */
    
  return 0;
}
