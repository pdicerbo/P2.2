#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/system_solvers.h"

#ifdef __MPI
#include <mpi.h>
#endif /* __MPI */

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
  double r_hat_square, alpha, beta, b_mod_square, r_mod_prev;
  
  /* arrays initialization */
  for(j = 0; j < N; j++){
    x[j] = 0.;
    r[j] = b[j];
    p[j] = b[j];
  }

  b_mod_square = vector_prod(b, b, N);
  
  *n_iter = 0;
  r_hat_square = vector_prod(r, r, N);
  r_hat_square /= b_mod_square;

  while(r_hat_square > prec * prec){
    t = mat_vec_prod(A, p, N);
    alpha = vector_prod(r, r, N);
    alpha /= vector_prod(p, t, N);

    r_mod_prev = vector_prod(r, r, N);
    
    for(j = 0; j < N; j++){
      x[j] += alpha * p[j];
      r[j] -= alpha * t[j];
    }

    beta = vector_prod(r, r, N) / r_mod_prev;

    for(j = 0; j < N; j++)
      p[j] = r[j] + beta * p[j];

    r_hat_square = vector_prod(r, r, N) / b_mod_square;
    free(t);

    (*n_iter)++;
  }

  free(r);
  free(p);
}

/* Perform the CONJUGATE GRADIENT ALGORITHM to obtain */
/* the solution of the system "A x = b" with relative error "prec"*/
/* This function store the result into array x */
/* The number of iterations is stored into n_iter */
/* the found solution is stored in x */
void conj_guess(double* A, double* x, double* b, double* guess, double prec, int N, int* n_iter){

  int j;
  double *r = (double*) malloc(N * sizeof(double));
  double *p = (double*) malloc(N * sizeof(double));
  double *t;
  double r_hat_square, alpha, beta, b_mod_square, r_mod_prev;

  /* calculation of the A x_guess */
  t = mat_vec_prod(A, guess, N);
  
  /* arrays initialization */
  for(j = 0; j < N; j++){
    x[j] = 0.;
    r[j] = b[j] - t[j];
    p[j] = b[j] - t[j];
  }

  free(t);
  
  b_mod_square = vector_prod(p, p, N);
  
  *n_iter = 0;
  r_hat_square = vector_prod(r, r, N);
  r_hat_square /= b_mod_square;

  while(r_hat_square > prec * prec){
    t = mat_vec_prod(A, p, N);
    alpha = vector_prod(r, r, N);
    alpha /= vector_prod(p, t, N);

    r_mod_prev = vector_prod(r, r, N);
    
    for(j = 0; j < N; j++){
      x[j] += alpha * p[j];
      r[j] -= alpha * t[j];
    }

    beta = vector_prod(r, r, N) / r_mod_prev;

    for(j = 0; j < N; j++)
      p[j] = r[j] + beta * p[j];

    r_hat_square = vector_prod(r, r, N) / b_mod_square;
    free(t);

    (*n_iter)++;
  }

  /* "removing" x_guess: x = x_guess + \Delta x */
  for(j = 0; j < N; j++)
    x[j] += guess[j];
    
  free(r);
  free(p);
}

/* Perform the CONJUGATE GRADIENT ALGORITHM with the **sparse_prod** function to obtain */
/* the solution of the system "A x = b" with relative error "prec"*/
/* This function store the result into array x */
/* The number of iterations is stored into n_iter */
/* the found solution is stored in x */
/* void sparse_conj_grad_alg(double* A, double* x, double* b, double prec, int N, int* n_iter){ */
#ifdef __MPI
void sparse_conj_grad_alg(double* x, double* b, double sigma, double s, double prec, int N, int* n_iter, int MyID, int NPE){
#else
void sparse_conj_grad_alg(double* x, double* b, double sigma, double s, double prec, int N, int* n_iter){
#endif
  int j;
  double *r = (double*) malloc(N * sizeof(double));
  double *p = (double*) malloc(N * sizeof(double));
  double *t;
  double r_hat_square, alpha, beta, b_mod_square, r_mod_prev;
  
  /* arrays initialization */
  for(j = 0; j < N; j++){
    x[j] = 0.;
    r[j] = b[j];
    p[j] = b[j];
  }

  b_mod_square = vector_prod(b, b, N);

  *n_iter = 0;
  r_hat_square = vector_prod(r, r, N) / b_mod_square;

  while(r_hat_square > prec * prec){
    
#ifdef __MPI

    t = sparse_prod(p, sigma, s, N, MyID, NPE);

#else
    
    t = sparse_prod(p, sigma, s, N);
    
#endif /* __MPI */
    alpha = vector_prod(r, r, N);
    alpha /= vector_prod(p, t, N);

    r_mod_prev = vector_prod(r, r, N);
    
    for(j = 0; j < N; j++){
      x[j] += alpha * p[j];
      r[j] -= alpha * t[j];
    }

    beta = vector_prod(r, r, N) / r_mod_prev;

    for(j = 0; j < N; j++)
      p[j] = r[j] + beta * p[j];

    r_hat_square = vector_prod(r, r, N) / b_mod_square;
    free(t);

    (*n_iter)++;
  }

  free(r);
  free(p);
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
#ifdef __MPI

double* sparse_prod(double* x, double sigma, double s, int N, int MyID, int NPE){

#else

double* sparse_prod(double* x, double sigma, double s, int N){

#endif /* __MPI */

  double* ret = (double*) calloc(N, sizeof(double));
  int i;

#ifdef __MPI
  int sendtag = 42, rectag = 24;
  
  /* computation of the product of the diagonal elements*/
  for(i = 0; i < N; i++)
    ret[i] = (sigma + 1.) * x[i];

  fprintf(stderr, "MyID = %d, NPE = %d, back = %d, next = %d\n", MyID, NPE, (MyID+NPE-1) % NPE, (MyID+1)% NPE);

    /* "up shifting" */
  MPI_Sendrecv(x, 1, MPI_DOUBLE, (MyID+NPE-1)%NPE, sendtag, x, 1, MPI_DOUBLE, (MyID+1)%NPE, rectag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  fprintf(stderr, "MyID = %d, back = %d, next = %d\n", MyID, (MyID+NPE-1) % NPE, (MyID+1) % NPE);

  /* updating the product */
  for(i = 0; i < N; i++)
    ret[i] += s * x[(i+1)%N];

  /* "arrow inversion" */
  MPI_Sendrecv(x, 1, MPI_DOUBLE, (MyID+1)%NPE, sendtag, x, 1, MPI_DOUBLE, (MyID+NPE-1)%NPE, rectag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  /* "down shifting" */
  MPI_Sendrecv(&(x[N-1]), 1, MPI_DOUBLE, (MyID+1 % NPE), sendtag, &(x[N-1]), 1, MPI_DOUBLE, (MyID+NPE-1 % NPE), rectag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  /* "finalizing" the product */
  for(i = 0; i < N; i++)
    ret[i] += s * x[(N+i-1)%N];

#else
  
  /* first and last entry of the vector are computed "by hand" */
  ret[0] = (sigma + 1.) * x[0] + s * x[1] + s * x[N - 1];
  
  for(i = 1; i < N - 1; i++)
    ret[i] = s * x[i - 1] + (sigma + 1.) * x[i] + s * x[i + 1];

  ret[N - 1] = s * x[0] + s * x[N - 2] + (sigma + 1.) * x[N - 1];
  
#endif /* __MPI */
  
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
  double *p;
  double r_hat_square, alpha, beta, Fx;
  double r_e, mod_b_square, r_prev_mod;
  int j;
  FILE* min_check;
  FILE* explicit;

  r = (double*) malloc(N * sizeof(double));
  p = (double*) malloc(N * sizeof(double));
  r_explicit = (double*) malloc(N * sizeof(double));
  
  min_check = fopen("results/min_check_conj.dat", "w");
  explicit = fopen("results/explicit_res.dat", "w");
  
  /* arrays initialization */
  for(j = 0; j < N; j++){
    x[j] = 0.;
    r[j] = b[j];
    p[j] = b[j];
  }
  mod_b_square = vector_prod(b, b, N);
  
  *n_iter = 0;
  r_hat_square = vector_prod(r, r, N) / mod_b_square;

  while(r_hat_square > prec * prec){
    t = mat_vec_prod(A, p, N);
    alpha = vector_prod(r, r, N);
    alpha /= vector_prod(p, t, N);

    r_prev_mod = vector_prod(r, r, N);
    
    for(j = 0; j < N; j++){
      x[j] += alpha * p[j];
      r[j] -=  alpha * t[j];
    }

    beta = vector_prod(r, r, N) / r_prev_mod;

    for(j = 0; j < N; j++)
      p[j] = r[j] + beta * p[j];

    r_hat_square = vector_prod(r, r, N) / mod_b_square;

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
  free(r_explicit);
}
