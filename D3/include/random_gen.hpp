#ifndef _RANDOM_GEN_HPP 
#define _RANDOM_GEN_HPP

//fill a vector b[n] with random numbers (see below)
#ifdef  __cplusplus
extern "C"{
#endif
void fill_source(double *b,double ave,double sigma,int n);

//fill a matrix with random gaussianly distributed number with average "ave" and
//standard deviation "sigma"
void fill_gauss_matrix(double *m,double ave,double sigma,int n);

//fill a matrix m[n X n] with random numbers, imposing definite positiveness and 
//fixing condition number to "cond_numb"
void fill_defpos_symm_matrix(double *m,double cond_numb,int n);

//initialize the random generator
void random_init(int seed);
#ifdef __cplusplus
}
#endif
#endif
