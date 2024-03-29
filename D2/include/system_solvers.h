#ifndef __SYSTEM_SOLVERS
#define __SYSTEM_SOLVERS

#ifdef  __cplusplus
extern "C"{
#endif

void gradient_alg(double*, double*, double*, double, int, int*);
double vector_prod(double*, double*, int);
void mat_vec_prod(double*, double*, double*, int);
void sparse_prod(double*, double*, double, double, int);
void conj_grad_alg(double*, double*, double*, double, int, int*);
void sparse_conj_grad_alg(double*, double*, double, double, double, int, int*);
void inner_checks(double* , double*, double*, double, int, int*);

#ifdef  __cplusplus
}
#endif

#endif /* __SYSTEM_SOLVERS */
