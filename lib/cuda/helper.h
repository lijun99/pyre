// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

// code guard
#ifndef cudalib_helper_h
#define cudalib_helper_h

//preferred choices of number of threads in a block
// for processing 1D
#define NTHREADS 256
#define BLOCKDIM 256

// preferred block size for 2D
#define NTHREADS2D 16
#define BDIMX 16
#define BDIMY 16

// warp size
#define WARPSIZE 32
#define FULL_MASK 0xffffffff //for shuffle

// max threads per block
#define MAXTHREADS 1024

#ifdef __FERMI__ //2.0: M2090
#define MAXBLOCKSX 65535  //x
#define MAXBLOCKSYZ 65535 //y,z
#else //2.0 and above : K40, ...
#define MAXBLOCKSX 4294967295 //x
#define MAXBLOCKSYZ 65535  //y,z
#endif

// get index for 2d matrix element (i,j)
#define IDX2R(i,j,NJ) (((i)*(NJ))+(j))  //row-major order, c/c++/python
#define IDX2C(i,j,NI) (((j)*(NI))+(i))  //col-major order, Fortran/cublas
#define IDXDIAG(i, N) ((i)*N+i) //diagonal element


// ceil function
#define IDIVUP(i,j) ((i+j-1)/j)

#define IMUL(a, b) __mul24(a, b)

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#ifndef MIN
#define MIN(a,b) (a > b ? b: a)
#endif

#define PI 3.141592654f

#endif // cudalib_helper_h
//end of file
