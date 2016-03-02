#ifndef __SYSTEM_SOLVERS
#define __SYSTEM_SOLVERS

#ifdef  __cplusplus
extern "C"{
#endif

void gradient_alg(double*, double*, double*, double, int, int*);
double vector_prod(double*, double*, int);
double* mat_vec_prod(double*, double*, int);
#ifdef __MPI
double* sparse_prod(double*, double, double, int, int, int);
void sparse_conj_grad_alg(double*, double*, double, double, double, int, int*, int, int);
#else
double* sparse_prod(double*, double, double, int);
void sparse_conj_grad_alg(double*, double*, double, double, double, int, int*);
#endif
void conj_grad_alg(double*, double*, double*, double, int, int*);
void inner_checks(double* , double*, double*, double, int, int*);
void conj_guess(double*, double*, double*, double*, double, int, int*);

#ifdef  __cplusplus
}
#endif

#endif /* __SYSTEM_SOLVERS */