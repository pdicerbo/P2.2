#include <stdio.h>
#include <stdlib.h>
#include "../include/random_gen.hpp"
#include "../include/system_solvers.h"

int main(int argc, char** argv){

  double *A, *b, *sol, *check_sol;

  double r_hat = 1.e-8, cond_numb = 1.e5;
  
  int N = 2, n_iter, i, j;
  
  A = (double*) malloc(N * N * sizeof(double));
  b = (double*) malloc(N * sizeof(double));
  sol = (double*) malloc(N * sizeof(double));

  /* Filling matrix with the assignment values */
  A[0] = 3.;
  A[1] = 1.;
  A[2] = 1.;
  A[3] = 2.;

  b[0] = 1.;
  b[1] = 3.;

  printf("\n\tGRADIENT ALGORITHM\n");

  gradient_alg(A, sol, b, r_hat, N, &n_iter);

  check_sol = mat_vec_prod(A, sol, N);

  /* trivial solution control  */
  i = 0;
  for(j = 0; j < N; j++){
    if(abs(sol[j] - check_sol[j]) > 1.e10)
      i = 1;
  }

  if(i == 0)
    printf("\n\tThe found solution is correct\n\tResult obtained in %d iteration\n", n_iter);
  else
    printf("\n\tThe found solution is wrong\n");

  free(check_sol);
  
  printf("\n\n\tCONJUGATE GRADIENT ALGORITHM\n");
  
  conj_grad_alg(A, sol, b, r_hat, N, &n_iter);

  check_sol = mat_vec_prod(A, sol, N);

  /* trivial solution control */
  i = 0;
  for(j = 0; j < N; j++){
    if(abs(sol[j] - check_sol[j]) > 1.e10)
      i = 1;
  }

  if(i == 0)
    printf("\n\tThe found solution is correct\n\tResult obtained in %d iteration\n", n_iter);
  else
    printf("\n\tThe found solution is wrong\n");

  printf("\n");

  /* deallocate pointers */
  free(A);
  free(b);
  free(sol);
  free(check_sol);


  printf("\n\tstarting INNER CHECKS\n");
  N = 150;
  cond_numb = 1e5;
  r_hat = 1.e-28;
  A = (double*) malloc(N * N * sizeof(double));
  b = (double*) malloc(N * sizeof(double));
  sol = (double*) malloc(N * sizeof(double));
  fill_defpos_symm_matrix(A, cond_numb, N);
  fill_source(b, 2., 0.5, N);

  inner_checks(A, sol, b, r_hat, N, &n_iter);
  printf("\n\tDONE\n\n");

  free(A);
  free(b);
  free(sol);

  return 0;
}
