#include <stdio.h>
#include <stdlib.h>
#include "../include/random_gen.hpp"
#include "../include/system_solvers.h"
#include "../include/inverse_laplace_operator.hpp"
#include "../include/laplace_utils.h"

int main(int argc, char** argv){

  double *A, *A_shift, *b, *sol, *check_sol, *guess;
  double r_hat = 1.e-12, cond_numb = 1.e5, delta = 5., sigma = 0.6, s = -0.5;
  
  int N = 150, n_iter, i, j, offset;
  
  A = (double*) malloc(N * N * sizeof(double));
  b = (double*) malloc(N * sizeof(double));
  sol = (double*) malloc(N * sizeof(double));
  check_sol = (double*) malloc(N * sizeof(double));
  guess = (double*) malloc(N * sizeof(double));
  A_shift = (double*) malloc(N * N * sizeof(double));
  
  printf("\n\n\tINITIAL GUESS section\n");

  init_laplace_matrix(A, sigma, s, N);
  
  /* randomly filling b vector */
  fill_source(b, 2.2, 0.5, N);

  conj_grad_alg(A, guess, b, r_hat, N, &n_iter);

  offset = 0;
  for(j = 0; j < N*N; j += N){
    A_shift[j + offset] = A[j + offset] + delta;
    offset++;
  }
  
  conj_guess(A, sol, b, guess, r_hat, N, &n_iter);

  check_sol = mat_vec_prod(A, sol, N);

  i = 0;
  for(j = 0; j < N; j++){
    if(abs(sol[j] - check_sol[j]) > 1.e10){
      i = 1;
      break;
    }
  }

  if(i == 0)
    printf("\n\tThe found solution is correct\n\tResult obtained in %d iteration\n", n_iter);
  else
    printf("\n\tThe found solution is wrong\n");

  printf("\n");
  
  return 0;
}
