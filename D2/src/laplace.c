#include <stdio.h>
#include <stdlib.h>
#include "../include/inverse_laplace_operator.hpp"
#include "../include/random_gen.hpp"
#include "../include/system_solvers.h"

int main(){

  double *M, *f, *b, *check_sol;
  double r_hat = 1.e-8;
  double sigma = 0.6, D = 1., s = -0.5;
  int i, j, n_it;

  //space size
  int L = 6;

  M = (double*) malloc(L * L * sizeof(double));
  f = (double*) malloc(L * sizeof(double));
  b = (double*) malloc(L * sizeof(double));
  check_sol = (double*) malloc(L * sizeof(double));
  
  for(i = 0; i < L; i++){
    for(j = 0; j < L; j++){
      if(i == j)
	M[j + i*L] = sigma + D;
      else if(abs(i-j) == 1)
	M[j + i*L] = s;
      else
	M[j + i*L] = 0.;
    }
  }

  M[L - 1] = s;
  M[L * (L - 1)] = s;

  /* randomly filling b vector */
  fill_source(b, 2.2, 0.5, L);

  conj_grad_alg(M, f, b, r_hat, L, &n_it);

  inverse_laplace_operator(check_sol, b, sigma, L, L);
  /* print the matrix */
  for(i = 0; i < L; i++){
    for(j = 0; j < L; j++)
      printf("\t%lg", M[j + i*L]);
    printf("\n");
  }

  printf("\n\tMy solution:    Check_sol:\n");
  
  for(j = 0; j < L; j++)
    printf("\t%lg\t\t%lg\n", f[j], check_sol[j]);
  
  sparse_conj_grad_alg(M, f, b, r_hat, L, &n_it);

  printf("\n\tSp solution:    Check_sol:\n");
  
  for(j = 0; j < L; j++)
    printf("\t%lg\t\t%lg\n", f[j], check_sol[j]);

  free(M);
  free(f);
  free(b);
  free(check_sol);
  
  return 0;
}
