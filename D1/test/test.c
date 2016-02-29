#include <stdio.h>
#include <stdlib.h>
#include "../include/random_gen.hpp"
#include "../include/system_solvers.h"

int main(int argc, char** argv){

  /* printf("\n\tHello World\n"); */

  double* A;
  double* b;
  double* sol;
  double* check_sol;
  double r_hat = 1.e-8;
  
  int N = 2, n_iter;
  
  A = (double*) malloc(N * N * sizeof(double));
  b = (double*) malloc(N * sizeof(double));
  sol = (double*) malloc(N * sizeof(double));
  check_sol = (double*) malloc(N * sizeof(double));
  /* Filling matrix with the assignment values */
  A[0] = 3.;
  A[1] = 1.;
  A[2] = 1.;
  A[3] = 2.;

  b[0] = 1.;
  b[1] = 3.;

  check_sol[0] = -1. / 5.;
  check_sol[1] = 8. / 5.;
    
  gradient_alg(A, sol, b, r_hat, N, &n_iter);

  if(abs(check_sol[0] - sol[0]) < r_hat && abs(check_sol[1] - sol[1]) < r_hat){
    printf("\n\tGRADIENT ALGORITHM");
    printf("\n\n\tTEST PASSED!");
  }
  else{
    printf("\n\tGRADIENT ALGORITHM");
    printf("\n\n\tTEST NOT PASSED\n");
    printf("\tdifference on x = %lg; difference on y = %lg", check_sol[0] - sol[0], check_sol[1] - sol[1]);
  }

  printf("\n\n\n");

  conj_grad_alg(A, sol, b, r_hat, N, &n_iter);

  r_hat = 1.e-15;
  
  if(abs(check_sol[0] - sol[0]) < r_hat && abs(check_sol[1] - sol[1]) < r_hat){
    printf("\n\tCONJUGATE GRADIENT ALGORITHM");
    printf("\n\n\tTEST PASSED!");
  }
  else{
    printf("\n\tGRADIENT ALGORITHM");
    printf("\n\n\tTEST NOT PASSED\n");
    printf("\tdifference on x = %lg; difference on y = %lg", check_sol[0] - sol[0], check_sol[1] - sol[1]);
  }

  printf("\n\n\n");

  /* deallocate pointers */
  free(A);
  free(b);
  free(sol);
  free(check_sol);
  
  return 0;
}
