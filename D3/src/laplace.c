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
  
  L = 6; // vector size

#ifdef __MPI

  int NPE, MyID, MyTag, vsize, rest, l_tmp;
  double* b_send;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &MyID);
  MPI_Comm_size(MPI_COMM_WORLD, &NPE);

  /* printf("MyID = %d, s = %lg, L = %d", MyID, s, L); */
  MyTag = 42;
  vsize = L;
  L /= NPE;
  rest = vsize % NPE;

  if(rest != 0 && MyID < rest)
    L++;

#endif /* __MPI */  

  f = (double*) malloc(L * sizeof(double));
  b = (double*) malloc(L * sizeof(double));

#ifdef __MPI

  if(MyID == 0){
    /* process 0 generates the random b vector and sends */
    /* various pieces to the other processes */
    b_send = (double*) malloc(L * sizeof(double));

    /* "segment" of the b vector that belong to process 0 */
    fill_source(b, 2.2, 0.5, L);

    /* need use l_tmp because if rest != 0, process 0 have to send bunch */
    /* of array of different size */			      
    l_tmp = L;

    for(j = 1; j < NPE; j++){
      
      if(rest != 0 && j == rest)
	l_tmp--;
      
      fill_source(b_send, 2.2, 0.5, l_tmp);

      MPI_Send(b_send, l_tmp, MPI_DOUBLE, j, MyTag, MPI_COMM_WORLD);
      
    }
  }
  else{
    /* processes with MyID != 0 receives alwais in b */
    MPI_Recv(b, L, MPI_DOUBLE, 0, MyTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

#else
  
  /* randomly filling b vector */
  fill_source(b, 2.2, 0.5, L);

#endif /* __MPI */

  sparse_conj_grad_alg(f, b, sigma, s, r_hat, L, &n_it);
  
#ifdef __MPI
  if(MyID == 0){
    
  check_sol = (double*) malloc(vsize * sizeof(double));

#else

  check_sol = (double*) malloc(L * sizeof(double));

#endif /* __MPI */
  
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
#ifdef __MPI
  }
#endif /* __MPI */

#ifdef __MPI
  
  /* for(i = 0; i < L; i++) */
  /*   printf("\t%d\t%d\n", i, (i+1) % L); */
  /* printf("\n\n"); */
  /* for(i = 0; i < L; i++) */
  /*   printf("\t%d\t%d\n", i, (L + i - 1) % L); */

  MPI_Finalize();
  
#endif /* __MPI */
  
  free(f);
  free(b);

#ifdef __MPI
  if(MyID == 0){
    free(b_send);
#endif /* __MPI */

  free(check_sol);
  
#ifdef __MPI
  }
#endif /* __MPI */
    
  return 0;
}
