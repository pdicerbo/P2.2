#include <stdio.h>
#include <stdlib.h>
#include "../include/system_solvers.h"

/* Perform the GRADIENT ALGORITHM to obtain */
/* the solution of the system "A x = b" with relative error "prec"*/
/* This function store the result into array x */
/* The number of iterations is stored into n_iter */
void gradient_alg(double* A, double* x, double* b, double prec, int N, int* n_iter){

  double* r; /* residue error */
  double* t; /* array in which store A r_(k - 1) */
  double r_hat_square, alpha;
  int j;

  r = (double*) malloc(N * sizeof(double));

  for(j = 0; j < N; j++){
    x[j] = 0.;
    r[j] = b[j];
  }

  *n_iter = 0;
  r_hat_square = vector_prod(r, r, N);
  r_hat_square /= vector_prod(b, b, N);

  /* avoid the square root calculation in the condition within the while */
  while(r_hat_square > prec * prec){
    t = mat_vec_prod(A, r, N);
    alpha = vector_prod(r, r, N);
    alpha /= vector_prod(r, t, N);

    for(j = 0; j < N; j++){
      x[j] += alpha * r[j];
      r[j] -= alpha * t[j];
    }

    /* perform the calculation of the new relative error */
    r_hat_square = vector_prod(r, r, N);
    r_hat_square /= vector_prod(b, b, N);
    (*n_iter)++;
    free(t);
  }

  /* deallocate pointers */
  free(r);  
}

/* Perform the CONJUGATE GRADIENT ALGORITHM to obtain */
/* the solution of the system "A x = b" with relative error "prec"*/
/* This function store the result into array x */
/* The number of iterations is stored into n_iter */
void conj_grad_alg(double* A, double* x, double* b, double prec, int N, int* n_iter){

  int j;
  double *r = (double*) malloc(N * sizeof(double));
  double *p = (double*) malloc(N * sizeof(double));
  double *t;
  double *r_old = (double*) malloc(N * sizeof(double));
  double r_hat_square, alpha, beta;
  
  /* arrays initialization */
  for(j = 0; j < N; j++){
    x[j] = 0.;
    r_old[j] = b[j];
    p[j] = b[j];
  }

  *n_iter = 0;
  r_hat_square = vector_prod(r_old, r_old, N);
  r_hat_square /= vector_prod(b, b, N);

  while(r_hat_square > prec * prec){
    t = mat_vec_prod(A, p, N);
    alpha = vector_prod(r_old, r_old, N);
    alpha /= vector_prod(p, t, N);

    for(j = 0; j < N; j++){
      x[j] += alpha * p[j];
      r[j] = r_old[j] - alpha * t[j];
    }

    beta = vector_prod(r, r, N);
    beta /= vector_prod(r_old, r_old, N);

    for(j = 0; j < N; j++){
      p[j] = r[j] + beta * p[j];
      r_old[j] = r[j];
    }
    r_hat_square = vector_prod(r_old, r_old, N);
    r_hat_square /= vector_prod(b, b, N);
    free(t);

    (*n_iter)++;
  }

  free(r);
  free(p);
  free(r_old);
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

void minimization_check(double* A, double* x, double* b, double prec, int N, int* n_iter){

  double* r; /* residue error */
  double* t; /* array in which store A r_(k - 1) */
  double* tmp;
  double *r_old;
  double *p;
  double r_hat_square, alpha, beta, Fx;
  int j;
  FILE* min_check;

  r = (double*) malloc(N * sizeof(double));

  for(j = 0; j < N; j++){
    x[j] = 0.;
    r[j] = b[j];
  }

  *n_iter = 0;
  r_hat_square = vector_prod(r, r, N);
  r_hat_square /= vector_prod(b, b, N);

  min_check = fopen("results/min_check_grad.dat", "w");
  
  /* avoid the square root calculation in the condition within the while */
  while(r_hat_square > prec * prec){
    t = mat_vec_prod(A, r, N);
    alpha = vector_prod(r, r, N);
    alpha /= vector_prod(r, t, N);

    for(j = 0; j < N; j++){
      x[j] += alpha * r[j];
      r[j] -= alpha * t[j];
    }
      
    /* perform the calculation of the new relative error */
    r_hat_square = vector_prod(r, r, N);
    r_hat_square /= vector_prod(b, b, N);
    (*n_iter)++;

    /* calculation of F(x) = 0.5 * x^T A x - b x */
    tmp = mat_vec_prod(A, x, N);
    Fx = 0.5 * vector_prod(x, tmp, N);
    Fx -= vector_prod(b, x, N);

    fprintf(min_check, "%d\t%lg\n", *n_iter, Fx);
    
    free(t);
    free(tmp);
  }

  fclose(min_check);

  /* Second check */
  p = (double*) malloc(N * sizeof(double));
  r_old = (double*) malloc(N * sizeof(double));

  min_check = fopen("results/min_check_conj.dat", "w");
  
  /* arrays initialization */
  for(j = 0; j < N; j++){
    x[j] = 0.;
    r_old[j] = b[j];
    p[j] = b[j];
  }

  *n_iter = 0;
  r_hat_square = vector_prod(r_old, r_old, N);
  r_hat_square /= vector_prod(b, b, N);

  while(r_hat_square > prec * prec){
    t = mat_vec_prod(A, p, N);
    alpha = vector_prod(r_old, r_old, N);
    alpha /= vector_prod(p, t, N);

    for(j = 0; j < N; j++){
      x[j] += alpha * p[j];
      r[j] = r_old[j] - alpha * t[j];
    }

    beta = vector_prod(r, r, N);
    beta /= vector_prod(r_old, r_old, N);

    for(j = 0; j < N; j++){
      p[j] = r[j] + beta * p[j];
      r_old[j] = r[j];
    }
    r_hat_square = vector_prod(r_old, r_old, N);
    r_hat_square /= vector_prod(b, b, N);

    (*n_iter)++;
    /* calculation of F(x) = 0.5 * x^T A x - b x */
    tmp = mat_vec_prod(A, x, N);
    Fx = 0.5 * vector_prod(x, tmp, N);
    Fx -= vector_prod(b, x, N);

    fprintf(min_check, "%d\t%lg\n", *n_iter, Fx);
    
    free(t);
    free(tmp);
  }
  
  fclose(min_check);

  free(r);
  free(p);
  free(r_old);
}
