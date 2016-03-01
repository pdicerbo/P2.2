#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

void init_laplace_matrix(double* M, double sigma, double s, int L){

  double D = 1.;
  int i, j;


  for(i = 0; i < L; i++){
    for(j = 0; j < L; j++){
      if(i == j)
	M[j + i*L] = sigma + D;
      else if(abs(i-j) == 1)
	M[j + i*L] = s;
      else
	M[j + i*L] = 0.;
    }
  }
  
  M[L - 1] = s;
  M[L * (L - 1)] = s;
  
}

double seconds(){
  /* Return the second elapsed since Epoch (00:00:00 UTC, January 1, 1970) */
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}
