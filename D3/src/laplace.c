#include <stdio.h>
#include <stdlib.h>
#include "../include/inverse_laplace_operator.hpp"
#include "../include/random_gen.hpp"
#include "../include/system_solvers.h"
#include "../include/laplace_utils.h"

int main(){

  double *M, *f, *b, *check_sol;
  double r_hat = 1.e-15;
  double sigma = 0.6, s = -0.5;
  int i, j, n_it, L;

  /* double *eigenv; */
  /* double t_start, t_end; */
  /* int n_rep = 1000; */
  /* int L_start = 10, L_end = 5e2, L_step = 50; */
  /* FILE* classic; */
  /* FILE* sparse; */

  L = 6;
  M = (double*) malloc(L * L * sizeof(double));
  f = (double*) malloc(L * sizeof(double));
  b = (double*) malloc(L * sizeof(double));
  check_sol = (double*) malloc(L * sizeof(double));

  init_laplace_matrix(M, sigma, s, L);
    
  /* randomly filling b vector */
  fill_source(b, 2.2, 0.5, L);

  conj_grad_alg(M, f, b, r_hat, L, &n_it);

  inverse_laplace_operator(check_sol, b, sigma, L, L);

  printf("\n\tMy solution:    Check_sol:\n");

  i = 0;
  for(j = 0; j < L; j++){
    printf("\t%lg\t\t%lg\n", f[j], check_sol[j]);
    if(abs(f[j] - check_sol[j]) > 1.e-10)
      i = 1;
  }

  if(i == 0)
    printf("\n\tThe found solution is correct\n");
  else
    printf("\n\tThe found solution is wrong\n");
  
  sparse_conj_grad_alg(f, b, sigma, s, r_hat, L, &n_it);

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
  
  free(M);
  free(f);
  free(b);
  free(check_sol);
  
  /* fprintf(stderr, "\n\tTIMING SECTION\n"); */
  /* fprintf(stderr, "\tperform %d repetition for each matrix size\n\n", n_rep); */
  
  /* classic = fopen("results/classic_timing.dat", "w"); */
  /* fprintf(stderr, "\tSIZE\ttime (s)\n\n", L, t_end - t_start); */
  /* for(L = L_start; L < L_end; L += L_step){ */
  /*   M = (double*) malloc(L * L * sizeof(double)); */
  /*   f = (double*) malloc(L * sizeof(double)); */
  /*   b = (double*) malloc(L * sizeof(double)); */

    
  /*   init_laplace_matrix(M, sigma, s, L); */
    
  /*   /\* randomly filling b vector *\/ */
  /*   fill_source(b, 2.2, 0.5, L); */

  /*   t_start = seconds(); */

  /*   for(j = 0; j < n_rep; ++j) */
  /*     conj_grad_alg(M, f, b, r_hat, L, &n_it); */

  /*   t_end = seconds(); */

  /*   fprintf(classic, "%d\t%lg\n", L, t_end - t_start); */
  /*   fprintf(stderr, "\t%d\t%lg\n", L, t_end - t_start); */
    
  /*   free(M); */
  /*   free(f); */
  /*   free(b); */
  /* } */
  
  /* fclose(classic); */

  /* /\* Timing of the "sparse" optimized function *\/ */
  /* fprintf(stderr, "\n\tsparse section\n\n"); */
  /* sparse = fopen("results/sparse_timing.dat", "w"); */

  /* for(L = L_start; L < L_end; L += L_step){ */
  /*   M = (double*) malloc(L * L * sizeof(double)); */
  /*   f = (double*) malloc(L * sizeof(double)); */
  /*   b = (double*) malloc(L * sizeof(double)); */
    
  /*   init_laplace_matrix(M, sigma, s, L); */
    
  /*   /\* randomly filling b vector *\/ */
  /*   fill_source(b, 2.2, 0.5, L); */

  /*   t_start = seconds(); */

  /*   for(j = 0; j < n_rep; ++j) */
  /*     sparse_conj_grad_alg(f, b, sigma, s, r_hat, L, &n_it); */

  /*   t_end = seconds(); */

  /*   fprintf(sparse, "%d\t%lg\n", L, t_end - t_start); */
  /*   fprintf(stderr, "\t%d\t%lg\n", L, t_end - t_start); */
    
  /*   free(M); */
  /*   free(f); */
  /*   free(b); */
  /* } */
  
  /* fclose(sparse); */

  /* /\* CHECK CONDITION NUMBER *\/ */
  /* L = 6; */
  /* eigenv = (double*) malloc(L * sizeof(double)); */
  
  /* compute_eigenvalues(eigenv, sigma, L); */

  /* printf("\n\tPrinting the eigenvalues for matrix with size %d and sigma = %lg\n\n", L, sigma); */
  /* for(j = 0; j < L; j++) */
  /*   printf("\t%lg\n", eigenv[j]); */

  /* printf("\n"); */
  /* free(eigenv); */
  
  return 0;
}
