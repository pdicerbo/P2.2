#ifndef __SYSTEM_SOLVERS
#define __SYSTEM_SOLVERS

void gradient_alg(double*, double*, double*, double, int, int*);
double vector_prod(double*, double*, int);
double* mat_vec_prod(double*, double*, int);

#endif /* __SYSTEM_SOLVERS */
