#ifndef _FORTRAN_H_
#define _FORTRAN_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define FC_FUNC(lower,UPPER) lower##_

typedef struct { float real, imag; } scomplex_f;
typedef struct { double real, imag; } dcomplex_f;

#define MIN(a,b) ((b) < (a) ? (b) : (a))
#define MAX(a,b) ((a) < (b) ? (b) : (a))

#ifdef __cplusplus

#include <complex>

#define creal real
#define crealf real
#define cimag imag
#define cimagf imag

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

#define MAKE_COMPLEX(type,r,c) std::complex<type>(r,c)

#elif __STDC_VERSION__ >= 199901L

#include <complex.h>

typedef float complex scomplex;
typedef double complex dcomplex;

#define MAKE_COMPLEX(type,r,c) ((r)+(c)*I)

#else

#error "No complex type support."

#endif

#ifdef INT64
typedef int64_t integer;
#else
typedef int32_t integer;
#endif

typedef integer logical;

#endif
