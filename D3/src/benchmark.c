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
  double sigma = 2., s = -0.5;
  int i, j, n_it, L;
  
  L = 120000; // "full" vector size

#ifdef __MPI

  /* Initialization of the variables needed in the parallelized */
  /* version; each process works on a vector of size L / NumberProcessingElements */
  /* (unless there is a rest, that requires a work redistribution) */
  int NPE, MyID, MyTag, vsize, rest, l_tmp, count, repet = 1; //00;
  double* results_recv;
  int *displ, *recv;
  double tstart, tend;
  FILE* timing;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &MyID);
  MPI_Comm_size(MPI_COMM_WORLD, &NPE);

  /* FOR WEAK SCALING MEASURE */
  L *= NPE;

  MyTag = 42;
  vsize = L;
  L /= NPE;
  rest = vsize % NPE;

  if(rest != 0 && MyID < rest)
    L++;

  /* "initialization" section */
  if(MyID == 0){
    f = (double*) malloc(L * sizeof(double));
    b = (double*) malloc(vsize * sizeof(double));

    /* process 0 generates the random b vector and sends */
    /* various pieces to the other processes */
    /* while I send data, I initialize also "displs" and "recv" array */
    /* needed to gather the results */
    displ  = (int*) malloc(NPE * sizeof(int));
    recv   = (int*) malloc(NPE * sizeof(int));

    recv[0] = L;
    displ[0] = 0;

    /* process 0 stores all the b vector in order to check */
    /* the correctness of the results via "inverse_laplace_operator" function */
    /* otherwais, when needed, I could also gather all the pieces from the others processes */
    fill_source(b, 2.2, 0.5, vsize);

    /* need use l_tmp because if rest != 0, process 0 have to send bunch */
    /* of array of different size */			      
    l_tmp = L;
    count = 0;
    
    for(j = 1; j < NPE; j++){
      
      count += l_tmp;      

      if(rest != 0 && j == rest)
	l_tmp--;
      
      MPI_Send(&b[count], l_tmp, MPI_DOUBLE, j, MyTag, MPI_COMM_WORLD);
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

  for(j = 0; j < repet; ++j)
    sparse_conj_grad_alg(f, b, sigma, s, r_hat, L, &n_it, MyID, NPE);

  tend = MPI_Wtime();

  /* process 0 gather the results and checks the correctness */
  MPI_Gatherv(f, L, MPI_DOUBLE, results_recv, recv, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  if(MyID == 0){
    /* STRONG SCALING DATA */
  /* timing = fopen("results/strong_timing.dat", "a");     */
  /* fprintf(timing, "%d\t%lg\n", NPE, tend - tstart); */

    /* WEAK SCALING */
  timing = fopen("results/weak_timing.dat", "a");
  fprintf(timing, "%d\t%lg\n", vsize, tend - tstart);

  fclose(timing);
  /* check_sol = (double*) malloc(vsize * sizeof(double)); */
  /* inverse_laplace_operator(check_sol, b, sigma, vsize, vsize); */

  /* /\* printf("\n\tSparse solution:  Check_sol:\n"); *\/ */
  /* i = 0; */

  /* for(j = 0; j < vsize; j++){ */
  /*   /\* printf("\t%lg\t\t  %lg\n", results_recv[j], check_sol[j]); *\/ */
  /*   if(abs(results_recv[j] - check_sol[j]) > 1.e-6) */
  /*     i = 1; */
  /* } */

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
  
#endif /* __MPI */

  /* going to conclusion... */
  free(f);
  free(b);

#ifdef __MPI

  if(MyID == 0){
    free(results_recv);
    free(displ);
    free(recv);
#endif /* __MPI */

  free(check_sol);
  
#ifdef __MPI
  }
  
  MPI_Finalize();
  
#endif /* __MPI */
    
  return 0;
}
