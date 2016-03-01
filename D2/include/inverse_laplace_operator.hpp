#ifndef _INVERSE_LAPLACE_OPERATOR_HPP
#define _INVERSE_LAPLACE_OPERATOR_HPP

/* 
   The function 
   
   void inverse_laplace_operator(T1 *out,T1 *in,T2 sigma,int L,int V);
   
   Apply the inverse laplace operator on "in" vector and store it in "out"
   Sigma is the diagonal term
   in 1 dimension V=L=number of sites
*/

//specify the number of dimension - keep 1
#define NDIMS 1

//undef if compiling and linking with fftw3
//#define FFTW3

#include <cmath>
#include <complex>
#include <string.h>
#ifdef FFTW3
 #include <fftw3.h>
#endif
#include <iostream>

using namespace std;

typedef int coords[NDIMS];
#if NDIMS == 1
inline void coords_of_site(int *out,int in)
{out[0]=in;}
#endif

//used to generalize Fourier transform
template <class T> void sumassign(T &out,complex<T> in){out+=in.real();}
template <class T> void sumassign(complex<T> &out,complex<T> in){out+=in;}

//take Fourier transform over all momenta
template <class T1,class T2> void Fourier_transform(T1 *out,T2 *in,int sign,int L,int V)
{
  for(int k=0;k<V;k++)
    {
      //take momentum coordinates
      coords ck;
      coords_of_site(ck,k);
      
      //Fourier transform single momentum
      out[k]=0;
      for(int x=0;x<V;x++)
	{
	  //take coordinates of site
	  coords cx;
	  coords_of_site(cx,x);
	  
	  //compute argument of exponential
	  double arg=0;
	  for(int mu=0;mu<NDIMS;mu++) arg+=cx[mu]*ck[mu];
	  arg*=2*M_PI/L;
	  
	  sumassign(out[k],complex<double>(cos(arg),sign*sin(arg))*in[x]);
	}
      if(sign==-1) out[k]/=V;
    }
}

#ifdef FFTW3
inline void Fourier_transform(complex<double> *out,double *in,int sign,int L,int V)
{
  int sizes[NDIMS];
  for(int mu=0;mu<NDIMS;mu++) sizes[mu]=L;
  
  //copy to complex
  complex<double> *tempin=new complex<double>[V];
  for(int i=0;i<V;i++) tempin[i]=in[i];
  
  //take fftw
  fftw_plan plan=fftw_plan_dft(NDIMS,sizes,(fftw_complex*)tempin,(fftw_complex*)out,FFTW_FORWARD,FFTW_ESTIMATE);
  fftw_execute(plan);
  
  //delete
  fftw_destroy_plan(plan);
  delete[] tempin;
}
inline void Fourier_transform(double *out,complex<double> *in,int sign,int V)
{
  int sizes[NDIMS];
  for(int mu=0;mu<NDIMS;mu++) sizes[mu]=L;
  
  //perform fftw
  complex<double> *tempout=new complex<double>[V];
  fftw_plan plan=fftw_plan_dft(NDIMS,sizes,(fftw_complex*)in,(fftw_complex*)tempout,FFTW_BACKWARD,FFTW_ESTIMATE);
  fftw_execute(plan);
  
  //normalize
  for(int i=0;i<V;i++) out[i]=tempout[i].real()/V;
  
  //delete
  delete[] tempout;
  fftw_destroy_plan(plan);
}
#endif

//apply the inverse using an horrendous naive Fourier transform
template <class T1,class T2> void inverse_laplace_operator(T1 *out,T1 *in,T2 sigma,int L,int V)
{
  //allocate space for fftw
  complex<T1> *tilded=new complex<T1>[V];
  
  //take Fourier transform
  Fourier_transform(tilded,in,+1,L,V);
  
  //divide by Laplace operator in momentum space
  for(int k=0;k<V;k++)
    {
      //take momentum coordinates
      coords ck;
      coords_of_site(ck,k);
      
      //compute operator
      T1 op=sigma+NDIMS;
      for(int mu=0;mu<NDIMS;mu++) op-=cos(ck[mu]*2*M_PI/L);
      tilded[k]/=op;
    }
  
  //pass back to x space
  Fourier_transform(out,tilded,-1,L,V);
  delete [] tilded;
}

#endif
