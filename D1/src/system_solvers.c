#include <stdio.h>
#include <stdlib.h>
#include "../include/system_solvers.h"

/* Perform the GRADIENT ALGORITHM to obtain */
/* the solution of the system "A x = b" */
/* This function store the result into array x */
void gradient_alg(double* A, double* x, double* b, double prec, int N, int* n_iter){

  double* r; /* residue error */
  double* t; /* array in which store A r_(k - 1) */
  double r_hat, alpha;
  int j;

  r = (double*) malloc(N * sizeof(double));

  for(j = 0; j < N; j++){
    x[j] = 0.;
    r[j] = b[j];
  }

  *n_iter = 0;
  r_hat = vector_prod(r, r, N);
  r_hat /= vector_prod(b, b, N);

  while(r_hat > prec){
    t = mat_vec_prod(A, r, N);
    alpha = vector_prod(r, r, N);
    alpha /= vector_prod(r, t, N);

    for(j = 0; j < N; j++){
      x[j] += alpha * r[j];
      r[j] -= alpha * t[j];
    }


    /* perform the calculation of the following relative error */
    r_hat = vector_prod(r, r, N);
    r_hat /= vector_prod(b, b, N);
    (*n_iter)++;
  }

  free(t);
  free(r);
  
}

/* Perform the product between two vector of length N */
/* in this way: (x,y) = \sum_i x_i y_i */
double vector_prod(double* x, double* y, int N){

  double prod = 0.;
  int j;
  
  for(j = 0; j < N; j++)
    prod += x[j] * y[j];

  return prod;
}

/* Perform the product between a matrix A with size NxN */
/* and the vector x with length N in this way: */
/* (A x)_i = \sum_j A_ij x_j */
/* this function return a pointer to the memory area */
/* that contain the result of the operation (array of size N) */
double* mat_vec_prod(double* A, double* x, int N){

  double* ret = (double*) calloc(N, sizeof(double));
  int i, j;
  
  for(i = 0; i < N; i++)
    for(j = 0; j < N; j++)
      ret[i] += A[i*N + j] * x[j];

  return ret;
}
