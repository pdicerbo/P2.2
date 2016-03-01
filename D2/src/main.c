#include <stdio.h>
#include <stdlib.h>
#include "../include/random_gen.hpp"
#include "../include/system_solvers.h"

int main(int argc, char** argv){

  double* A;
  double* b;
  double* sol;
  double* check_sol;
  double r_hat = 1.e-10, cond_numb = 1.e6;
  
  int N = 2, n_iter;
  
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

  printf("\n\tsol_x = %lg, sol_y = %lg", sol[0], sol[1]);
  printf("\n\n\tCHECH sol:\n\tcheck_x = %lg, check_y = %lg\n", check_sol[0], check_sol[1]);
  printf("\n\tResult obtained in %d iteration", n_iter);

  free(check_sol);
  
  printf("\n\n\tCONJUGATE GRADIENT ALGORITHM\n");
  
  conj_grad_alg(A, sol, b, r_hat, N, &n_iter);

  check_sol = mat_vec_prod(A, sol, N);

  printf("\n\tsol_x = %lg, sol_y = %lg", sol[0], sol[1]);
  printf("\n\n\tCHECH sol:\n\tcheck_x = %lg, check_y = %lg\n", check_sol[0], check_sol[1]);
  printf("\n\tResult obtained in %d iteration\n", n_iter);

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
