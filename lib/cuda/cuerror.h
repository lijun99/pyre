// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

// code guard
#ifndef cudalib_error_h
#define cudalib_error_h

#include <portinfo> // for CUDA_DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <curand.h>
#include <cublas_v2.h>
#include <cusolverDn.h>


#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif


#ifdef CUDA_DEBUG
// error checking for cuda lib calls
#define cudaSafeCall(x) do { if((x) != cudaSuccess) { \
      printf("CUDA Error at %s:%d\n",__FILE__,__LINE__); \
      DEVICE_RESET \
      exit(EXIT_FAILURE);}} while(0)
#define curandSafeCall(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
      printf("CURAND Error at %s:%d\n",__FILE__,__LINE__); \
      DEVICE_RESET \
      exit(EXIT_FAILURE);}} while(0)
#define cublasSafeCall(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \
      printf("CUBLAS Error at %s:%d\n",__FILE__,__LINE__); \
      DEVICE_RESET \
      exit(EXIT_FAILURE);}} while(0)
#define cusolverSafeCall(x) do { if((x) != CUSOLVER_STATUS_SUCCESS) { \
      printf("CUSOLVER Error at %s:%d\n",__FILE__,__LINE__); \
      DEVICE_RESET \
      exit(EXIT_FAILURE);}} while(0)


// This will output the proper error string when calling cudaGetLastError
inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        DEVICE_RESET
        exit(EXIT_FAILURE);
    }
}
// error checking for customerized kernels
#define cudaCheckError(var)   __getLastCudaError (var, __FILE__, __LINE__)

#else //CUDA_DEBUG is not defined
#define cudaSafeCall(x) x
#define cublasSafeCall(x) x
#define curandSafeCall(x) x
#define cusolverSafeCall(x) x
#define cudaCheckError(x) 

#endif //CUDA_DEBUG

#endif //cudalib_error_h
// end of file
