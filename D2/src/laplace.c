#include <stdio.h>
#include <stdlib.h>
#include "../include/inverse_laplace_operator.hpp"
#include "../include/random_gen.hpp"
#include "../include/system_solvers.h"
#include "../include/laplace_utils.h"

int main(){

  double *M, *f, *b, *check_sol;
  double r_hat = 1.e-15;
  double sigma = 0.6, D = 1., s = -0.5;
  double t_start, t_end;
  int i, j, n_it, n_rep = 1000;
  int L, L_start = 10, L_end = 5e2, L_step = 50;
  FILE* classic;
  FILE* sparse;
  
  check_sol = (double*) malloc(L * sizeof(double));
  
  classic = fopen("results/classic_timing.dat", "w");
  
  for(L = L_start; L < L_end; L += L_step){
    M = (double*) malloc(L * L * sizeof(double));
    f = (double*) malloc(L * sizeof(double));
    b = (double*) malloc(L * sizeof(double));

    
    init_laplace_matrix(M, sigma, s, L);
    
    /* randomly filling b vector */
    fill_source(b, 2.2, 0.5, L);

    t_start = seconds();

    for(j = 0; j < n_rep; ++j)
      conj_grad_alg(M, f, b, r_hat, L, &n_it);

    t_end = seconds();

    fprintf(classic, "%d\t%lg\n", L, t_end - t_start);
    
    free(M);
    free(f);
    free(b);
  }
  
  fclose(classic);

  /* Timing of the "sparse" optimized function */
  sparse = fopen("results/sparse_timing.dat", "w");

  for(L = L_start; L < L_end; L += L_step){
    M = (double*) malloc(L * L * sizeof(double));
    f = (double*) malloc(L * sizeof(double));
    b = (double*) malloc(L * sizeof(double));
    
    init_laplace_matrix(M, sigma, s, L);
    
    /* randomly filling b vector */
    fill_source(b, 2.2, 0.5, L);

    t_start = seconds();

    for(j = 0; j < n_rep; ++j)
      sparse_conj_grad_alg(M, f, b, r_hat, L, &n_it);

    t_end = seconds();

    fprintf(sparse, "%d\t%lg\n", L, t_end - t_start);
    
    free(M);
    free(f);
    free(b);
  }
  
  fclose(sparse);

  free(check_sol);
  
  return 0;
}
