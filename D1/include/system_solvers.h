#ifndef __SYSTEM_SOLVERS
#define __SYSTEM_SOLVERS

void gradient_alg(double*, double*, double*, double, int, int*);
double vector_prod(double*, double*, int);
double* mat_vec_prod(double*, double*, int);
void conj_grad_alg(double*, double*, double*, double, int, int*);
void minimization_check(double* , double*, double*, double, int, int*);

#endif /* __SYSTEM_SOLVERS */
