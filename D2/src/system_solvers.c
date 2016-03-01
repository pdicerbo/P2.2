#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/system_solvers.h"

/* Perform the GRADIENT ALGORITHM to obtain */
/* the solution of the system "A x = b" with relative error "prec"*/
/* This function store the result into array x */
/* The number of iterations is stored into n_iter */
/* the found solution is stored in x */
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
/* the found solution is stored in x */
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

/* Perform the CONJUGATE GRADIENT ALGORITHM with the **sparse_prod** function to obtain */
/* the solution of the system "A x = b" with relative error "prec"*/
/* This function store the result into array x */
/* The number of iterations is stored into n_iter */
/* the found solution is stored in x */
void sparse_conj_grad_alg(double* A, double* x, double* b, double prec, int N, int* n_iter){

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
    t = sparse_prod(A, p, N);
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

/* Perform the product between a matrix and a vector */
/* taking into account that the matrix is sparse */
double* sparse_prod(double* A, double* x, int N){
  double* ret = (double*) calloc(N, sizeof(double));
  int i, j, offset = 0;

  /* first and last entry of the vector are computed "by hand" */
  ret[0] = A[0] * x[0] + A[1] * x[1] + A[N - 1] * x[N - 1];
  
  for(i = 1; i < N - 1; i++){
    for(j = 0; j < 3; j++){
      ret[i] += A[j + offset + i*N] * x[j +  offset];
    }
    offset++;
  }

  ret[N - 1] = A[N * (N - 1)] * x[0] + A[N * N - 2] * x[N - 2] + A[N * N - 1] * x[N - 1];

  return ret;
}

/* This function perform "inner" checks. The first is the calculation */
/* of the minimization of the functional (the last optional step of the D1 assignment). */
/* The second is the calculation of the explicit error in the conjugate gradient algorithm */
/* as required in the D2 assignment. Do **make x** and **make plot** into **D2** directory */
/* will produces the rispective plots in the **results** folder */
void inner_checks(double* A, double* x, double* b, double prec, int N, int* n_iter){

  double* r; /* residue error */
  double* r_explicit;
  double* t; /* array in which store A r_(k - 1) */
  double* tmp;
  double *r_old;
  double *p;
  double r_hat_square, alpha, beta, Fx;
  double r_e, mod_b_square;
  int j;
  FILE* min_check;
  FILE* explicit;

  r = (double*) malloc(N * sizeof(double));
  p = (double*) malloc(N * sizeof(double));
  r_old = (double*) malloc(N * sizeof(double));
  r_explicit = (double*) malloc(N * sizeof(double));
  
  min_check = fopen("results/min_check_conj.dat", "w");
  explicit = fopen("results/explicit_res.dat", "w");
  
  /* arrays initialization */
  for(j = 0; j < N; j++){
    x[j] = 0.;
    r_old[j] = b[j];
    p[j] = b[j];
  }
  mod_b_square = vector_prod(b, b, N);
  
  *n_iter = 0;
  r_hat_square = vector_prod(r_old, r_old, N);
  r_hat_square /= mod_b_square;

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
    r_hat_square /= mod_b_square; //vector_prod(b, b, N);

    (*n_iter)++;

    /* calculation of F(x) = 0.5 * x^T A x - b x */
    tmp = mat_vec_prod(A, x, N);
    Fx = 0.5 * vector_prod(x, tmp, N);
    Fx -= vector_prod(b, x, N);

    fprintf(min_check, "%d\t%lg\n", *n_iter, Fx);
    free(t);
    
    /* calculation of explicit residue */
    t = mat_vec_prod(A, x, N);

    // Storing in r_explicit directly the differenece between r_expl and r_impl 
    for(j = 0; j < N; j++)
      r_explicit[j] = b[j] - t[j];

    // Compute the modulus of the r_explicit vector and store it
    r_e = pow(vector_prod(r_explicit, r_explicit, N) / mod_b_square, 0.5);

    fprintf(explicit, "%d\t%lg\t%lg\n", *n_iter, r_e, pow(vector_prod(r, r, N) / mod_b_square, 0.5));
    
    free(t);
    free(tmp);
  }
  
  fclose(min_check);
  fclose(explicit);
  
  free(r);
  free(p);
  free(r_old);
  free(r_explicit);
}
