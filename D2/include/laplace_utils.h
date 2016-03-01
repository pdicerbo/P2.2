#ifndef __LAPLACE_UTILS_H
#define __LAPLACE_UTILS_H

#ifdef __cplusplus
extern "C"{
#endif
  
void init_laplace_matrix(double*, double, double, int);
double seconds();
void compute_eigenvalues(double*, double, int);
  
#ifdef __cplusplus
}
#endif

#endif
