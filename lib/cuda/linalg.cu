// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

// my declaration
#include "linalg.h"
// cuda utitlies
#include "cudalib.h"

#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;

namespace linalg_kernels {

    template<typename T, const int Nthreads> 
    __global__ void _determinant(T *g_idata, T *g_odata, unsigned int n);

    template<typename T, const int Nthreads> 
    __global__ void _logdet(T *g_idata, T *g_odata, unsigned int n);

}

// matrix inverse with LU
// float sepcialization
// to be compatible with gcc < 7, put specialization inside namespace 
namespace cudalib { namespace linalg {
template <>
int
inverse_cublas<float>(cublasHandle_t handle, float * const matrix, const size_t n, cudaStream_t stream)
{
    // set stream
    cublasSetStream(handle, stream);
    // one matrix
    int batchSize = 1;

    int *P, *INFO;

    cudaSafeCall(cudaMalloc(&P,n * batchSize * sizeof(int)));
    cudaSafeCall(cudaMalloc(&INFO,batchSize * sizeof(int)));

    int lda = n;

    float *A[] = { matrix };
    float** A_d;
    cudaSafeCall(cudaMalloc<float*>(&A_d,sizeof(A)));
    cudaSafeCall(cudaMemcpy(A_d,A,sizeof(A),cudaMemcpyHostToDevice));

    cublasSafeCall(cublasSgetrfBatched(handle,n,A_d,lda,P,INFO,batchSize));

    int INFOh = 0;
    cudaSafeCall(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh == n)
    {
        fprintf(stderr, "Factorization Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    cublasSafeCall(cublasSgetriBatched(handle,n,A_d,lda,P,A_d,lda,INFO,batchSize));

    cudaSafeCall(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh != 0)
    {
        fprintf(stderr, "Inversion Failed: Matrix is singular\n");
        cudaDeviceReset();
    }
    cudaFree(P), cudaFree(INFO), cudaFree(A_d);
    return INFOh;
} 

// double version
template <>
int
inverse_cublas<double>(cublasHandle_t handle, double * const matrix, const size_t n, cudaStream_t stream)
{

    cublasSetStream(handle, stream);

    int batchSize = 1;

    int *P, *INFO;

    cudaSafeCall(cudaMalloc(&P,n * batchSize * sizeof(int)));
    cudaSafeCall(cudaMalloc(&INFO,batchSize * sizeof(int)));

    int lda = n;

    double *A[] = { matrix };
    double** A_d;
    cudaSafeCall(cudaMalloc<double*>(&A_d,sizeof(A)));
    cudaSafeCall(cudaMemcpy(A_d,A,sizeof(A),cudaMemcpyHostToDevice));

    cublasSafeCall(cublasDgetrfBatched(handle,n,A_d,lda,P,INFO,batchSize));

    int INFOh = 0;
    cudaSafeCall(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh == n)
    {
        fprintf(stderr, "Factorization Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    cublasSafeCall(cublasDgetriBatched(handle,n,A_d,lda,P,A_d,lda,INFO,batchSize));

    cudaSafeCall(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh != 0)
    {
        fprintf(stderr, "Inversion Failed: Matrix is singular\n");
        cudaDeviceReset();
    }
    cudaFree(P), cudaFree(INFO), cudaFree(A_d);
    return INFOh;
}

template<>
int lu<double>(double * const matrix, const size_t lda, const size_t m, const size_t n, cudaStream_t stream)
{
    
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
    cusolverDnSetStream(solver_handle, stream);

    int work_size = 0;
    int *devInfo; 
    cudaSafeCall(cudaMalloc(&devInfo, sizeof(int)));

    int *devIpiv; //pivot indices
    cudaSafeCall(cudaMalloc(&devIpiv, min(m,n)*sizeof(int)));

    // get work_size
    // note that cusolver uses column-major, or m is leading dimension
    cusolverSafeCall(cusolverDnDgetrf_bufferSize(solver_handle,  m, n, matrix, lda, &work_size));

    // allocate work
    double *work;  
    cudaSafeCall(cudaMalloc(&work, work_size * sizeof(double)));

    // call LU
    cusolverSafeCall(cusolverDnDgetrf(solver_handle, m, n, matrix, lda, work, devIpiv, devInfo));
           
    // check error
    int info;
    cudaSafeCall(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    if(info!=0) {
        if(info <0)
            fprintf(stderr, "LU factorization error:  the %d-th parameter is wrong (not counting the handle)\n", -info);
        else
            fprintf(stderr, "LU factorization error:  U(%d, %d) =0 \n", info, info);
        cudaDeviceReset();
    }
    
    cudaSafeCall(cudaFree(work));
    cudaSafeCall(cudaFree(devInfo));
    cusolverSafeCall(cusolverDnDestroy(solver_handle));
    return info;
}

template<>
int lu<float>(float * const matrix, const size_t lda, const size_t m, const size_t n, cudaStream_t stream)
{
    
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
    cusolverDnSetStream(solver_handle, stream);

    int work_size = 0;
    int *devInfo; 
    cudaSafeCall(cudaMalloc(&devInfo, sizeof(int)));

    int *devIpiv; //pivot indices
    cudaSafeCall(cudaMalloc(&devIpiv, min(m,n)*sizeof(int)));

    // get work_size
    // note that cusolver uses column-major, or m is leading dimension
    cusolverSafeCall(cusolverDnSgetrf_bufferSize(solver_handle,  m, n, matrix, lda, &work_size));

    // 
    float *work;  
    cudaSafeCall(cudaMalloc(&work, work_size * sizeof(float)));
    cusolverSafeCall(cusolverDnSgetrf(solver_handle, m, n, matrix, lda, work, devIpiv, devInfo));
           
    // check error
    int info;
    cudaSafeCall(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    if(info!=0) {
        if(info <0)
            fprintf(stderr, "LU factorization error:  the %d-th parameter is wrong (not counting the handle)\n", -info);
        else
            fprintf(stderr, "LU factorization error:  U(%d, %d) =0 \n", info, info);
        cudaDeviceReset();
    }
    
    cudaSafeCall(cudaFree(work));
    cudaSafeCall(cudaFree(devInfo));
    cusolverSafeCall(cusolverDnDestroy(solver_handle));
    return info;
}

int
inverse_symm_D(double * matrix, cublasFillMode_t uplo, const size_t n, cudaStream_t stream)
{
    
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
    cusolverDnSetStream(solver_handle, stream);

    int n_int = (int)n;

    int work_size = 0;
    int *devInfo; 
    cudaSafeCall(cudaMalloc(&devInfo, sizeof(int)));

    // get work_size
    // note that cusolver uses column-major
    cusolverSafeCall(cusolverDnDpotrf_bufferSize(solver_handle, uplo, n, matrix, n, &work_size));

    // --- CUDA POTRF execution
    double *work;  
    cudaSafeCall(cudaMalloc(&work, work_size * sizeof(double)));
    cusolverSafeCall(cusolverDnDpotrf(solver_handle, uplo, n_int, matrix, n_int, work, work_size, devInfo));
    // check error
    int info;
    cudaSafeCall(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    if(info!=0) {
            fprintf(stderr, "Inverse by Chelosky factorization error:  the %d-th parameter is wrong (not counting the handle)\n", -info);
        cudaDeviceReset();
    }
    
    // to be safe, allocate a new work for cusolverDnDpotri
    int work_size2;
    cusolverSafeCall(cusolverDnDpotri_bufferSize(solver_handle, uplo, n, matrix, n, &work_size2));
    double * work2;
    cudaSafeCall(cudaMalloc(&work2, work_size2 * sizeof(double)));
    
    // call inverse 
    cusolverSafeCall(cusolverDnDpotri(solver_handle, uplo, n, matrix, n, work2, work_size2, devInfo));
    // check error
    cudaSafeCall(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    if(info!=0) {
        if(info <0)
            fprintf(stderr, "Chelosky inverse error:  the %d-th parameter is wrong (not counting the handle)\n", -info);
        else
            fprintf(stderr, "Chelosky inverse error:  the leading minor of order %d is not positive definite\n", info);
        cudaDeviceReset();
    }

    
    cudaSafeCall(cudaFree(work));
    cudaSafeCall(cudaFree(work2));
    cudaSafeCall(cudaFree(devInfo));
    cusolverSafeCall(cusolverDnDestroy(solver_handle));
    return info;
}

int
inverse_symm_S(float * matrix, cublasFillMode_t uplo, const size_t n, cudaStream_t stream)
{
    
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
    cusolverDnSetStream(solver_handle, stream);

    int work_size = 0;
    int *devInfo; 
    cudaSafeCall(cudaMalloc(&devInfo, sizeof(int)));

    int n_int = (int)n;

    // get work_size
    // note that cusolver uses column-major
    cusolverSafeCall(cusolverDnSpotrf_bufferSize(solver_handle, uplo, n, matrix, n, &work_size));

    // --- CUDA POTRF execution
    float *work;  
    cudaSafeCall(cudaMalloc(&work, work_size * sizeof(float)));
    cusolverSafeCall(cusolverDnSpotrf(solver_handle, uplo, n_int, matrix, n_int, work, work_size, devInfo));
    // check error
    int info;
    cudaSafeCall(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    if(info!=0) {
            fprintf(stderr, "Inverse by Chelosky factorization error:  the %d-th parameter is wrong (not counting the handle)\n", -info);
        cudaDeviceReset();
    }
    
        // to be safe, allocate a new work for cusolverDnDpotri
    int work_size2;
    cusolverSafeCall(cusolverDnSpotri_bufferSize(solver_handle, uplo, n, matrix, n, &work_size2));
    float * work2;
    cudaSafeCall(cudaMalloc(&work2, work_size2 * sizeof(float)));
    

    cusolverSafeCall(cusolverDnSpotri(solver_handle, uplo, n_int, matrix, n_int, work, work_size, devInfo));
    // check error
    cudaSafeCall(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    if(info!=0) {
        if(info <0)
            fprintf(stderr, "Chelosky inverse error:  the %d-th parameter is wrong (not counting the handle)\n", -info);
        else
            fprintf(stderr, "Chelosky inverse error:  the leading minor of order %d is not positive definite\n", info);
        cudaDeviceReset();
    }

    
    cudaSafeCall(cudaFree(work));
    cudaSafeCall(cudaFree(work2));
    cudaSafeCall(cudaFree(devInfo));
    cusolverSafeCall(cusolverDnDestroy(solver_handle));
    return info;
}

template<>
int
cholesky<double>(double * const matrix, cublasFillMode_t uplo, const size_t n, cudaStream_t stream)
{
    
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
    cusolverDnSetStream(solver_handle, stream);

    int work_size = 0;
    int *devInfo; 
    cudaSafeCall(cudaMalloc(&devInfo, sizeof(int)));

    // get work_size
    // note that cusolver uses column-major
    cusolverSafeCall(cusolverDnDpotrf_bufferSize(solver_handle, uplo, n, matrix, n, &work_size));

    // --- CUDA POTRF execution
    double *work;  
    cudaSafeCall(cudaMalloc(&work, work_size * sizeof(double)));
    cusolverSafeCall(cusolverDnDpotrf(solver_handle, uplo, n, matrix, n, work, work_size, devInfo));
    // check error
    int info;
    cudaSafeCall(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    if(info!=0) {
        if(info <0)
            fprintf(stderr, "Chelosky factorization error:  the %d-th parameter is wrong (not counting the handle)\n", -info);
        else
            fprintf(stderr, "Chelosky factorization error:  the leading minor of order %d is not positive definite\n", info);
        cudaDeviceReset();
    }
    cudaSafeCall(cudaFree(work));
    cudaSafeCall(cudaFree(devInfo));
    cusolverSafeCall(cusolverDnDestroy(solver_handle));
    return info;
}

template<>
int
cholesky<float>(float * const matrix, cublasFillMode_t uplo, const size_t n, cudaStream_t stream)
{
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
    cusolverDnSetStream(solver_handle, stream);

    int work_size = 0; 
    int *devInfo; 
    cudaSafeCall(cudaMalloc(&devInfo, sizeof(int)));

    // --- CUDA CHOLESKY initialization, get work_size
    // note that cusolver uses column-major
    cusolverSafeCall(cusolverDnSpotrf_bufferSize(solver_handle, uplo, n, matrix, n, &work_size));

    // --- CUDA POTRF execution
    float *work;  
    cudaSafeCall(cudaMalloc(&work, work_size * sizeof(float)));
    cusolverSafeCall(cusolverDnSpotrf(solver_handle, uplo, n, matrix, n, work, work_size, devInfo));
    // check error
    int info;
    cudaSafeCall(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

    if(info!=0) {
        if(info <0)
            fprintf(stderr, "Chelosky factorization error:  the %d-th parameter is wrong (not counting the handle)\n", -info);
        else
            fprintf(stderr, "Chelosky factorization error:  the leading minor of order %d is not positive definite\n", info);
        cudaDeviceReset();
    }
    cudaSafeCall(cudaFree(work));
    cudaSafeCall(cudaFree(devInfo));
    cusolverSafeCall(cusolverDnDestroy(solver_handle));
    return info;
}

// 
template<typename T>
T
determinant_triangular(T * const mat, const size_t n, cudaStream_t stream)
{
    int nthreads = NTHREADS;
    int nblocks = IDIVUP(n, 2*nthreads);
    dim3 blockSize (nthreads, 1, 1);
    dim3 gridSize  (nblocks, 1, 1);

    // create a work vector if n > nthreads 
    T *hprod = (T *) malloc(nblocks * sizeof(T));
    T *dprod = NULL;
    cudaSafeCall(cudaMalloc((void **) &dprod, nblocks * sizeof(T)));

    // reduce product in each block by gpu
    linalg_kernels::_determinant<T, NTHREADS><<<gridSize, blockSize, 0, stream>>>
        (mat, dprod, n);
    cudaCheckError("linalg_kernels::determinant error");
    
    cudaSafeCall(cudaMemcpy(hprod, dprod, nblocks * sizeof(T),cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaFree(dprod));

    T product = 1.0f;

    // reduce product for blocks
    for (int i = 0; i < nblocks; i++)
    {
        product *= hprod[i];
    }
    free(hprod);
    return product;
}

// explicit instantiation for shared library
template float determinant_triangular<float>(float * const, const size_t, cudaStream_t);
template double determinant_triangular<double>(double * const, const size_t, cudaStream_t);
template int determinant_triangular<int>(int * const, const size_t, cudaStream_t);

// 
template<typename T>
T
logdet_triangular(T * const mat, const size_t n, cudaStream_t stream)
{
    int nthreads = NTHREADS;
    int nblocks = IDIVUP(n, 2*nthreads);
    dim3 blockSize (nthreads, 1, 1);
    dim3 gridSize  (nblocks, 1, 1);

    // create a work vector if n > nthreads 
    T *hprod = (T *) malloc(nblocks * sizeof(T));
    T *dprod = NULL;
    cudaSafeCall(cudaMalloc((void **) &dprod, nblocks * sizeof(T)));

    // reduce product in each block by gpu
    linalg_kernels::_logdet<T, NTHREADS><<<gridSize, blockSize, 0, stream>>>
        (mat, dprod, n);
    cudaCheckError("linalg_kernels::determinant error");
    
    cudaSafeCall(cudaMemcpy(hprod, dprod, nblocks * sizeof(T),cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaFree(dprod));

    T product = 0.0f;

    // reduce product for blocks
    for (int i = 0; i < nblocks; i++)
    {
        product += hprod[i];
    }
    free(hprod);
    return product;
}

// explicit instantiation for shared library
template float logdet_triangular<float>(float * const, const size_t, cudaStream_t);
template double logdet_triangular<double>(double * const, const size_t, cudaStream_t);

template<typename T>
T determinant(T * const mat, const size_t n, cudaStream_t stream)
{
    // Cholesky decomposition
    int info = cudalib::linalg::cholesky<T>(mat, CUBLAS_FILL_MODE_LOWER, n, stream);
    // Use diagonal part 
    T product = cudalib::linalg::determinant_triangular<T>(mat, n, stream);
    return product*product;
}

// explicit instantiation for shared library
template float determinant<float>(float * const, const size_t, cudaStream_t);
template double determinant<double>(double * const, const size_t, cudaStream_t);


    } // of namespace linalg 
} // of ns cudalib


// the product of diagonal elements with shfl reduction
// adapted from NVIDIA_CUDA_SAMPLES reduce5

namespace linalg_kernels {
template<typename T, const int blockSize>
__global__ void
_determinant(T *g_idata, T *g_odata, unsigned int n)
{
       // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__  T sdata[blockSize];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

    T localProduct = (i < n) ? g_idata[IDXDIAG(i,n)] : 1;

    if (i + blockSize < n)
        localProduct *= g_idata[IDXDIAG(i+blockSize, n)];

    sdata[tid] = localProduct;
    cg::sync(cta);

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = localProduct = localProduct * sdata[tid + 256];
    }

    cg::sync(cta);

    if ((blockSize >= 256) &&(tid < 128))
    {
        sdata[tid] = localProduct = localProduct * sdata[tid + 128];
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = localProduct = localProduct * sdata[tid +  64];
    }

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    if (cta.thread_rank() < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) localProduct *= sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = tile32.size()/2; offset > 0; offset /= 2) 
        {
             localProduct *= tile32.shfl_down(localProduct, offset);
        }
    }

    // write result for this block to global mem
    if (cta.thread_rank() == 0) g_odata[blockIdx.x] = localProduct;
}


template<typename T, const int blockSize>
__global__ void
_logdet(T *g_idata, T *g_odata, unsigned int n)
{
       // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__  T sdata[blockSize];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

    T localSum = (i < n) ? log(g_idata[IDXDIAG(i,n)]) : 0;

    if (i + blockSize < n)
        localSum += log(g_idata[IDXDIAG(i+blockSize, n)]);

    sdata[tid] = localSum;
    cg::sync(cta);

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = localSum = localSum + sdata[tid + 256];
    }

    cg::sync(cta);

    if ((blockSize >= 256) &&(tid < 128))
    {
        sdata[tid] = localSum = localSum + sdata[tid + 128];
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = localSum = localSum + sdata[tid +  64];
    }

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    if (cta.thread_rank() < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) localSum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = tile32.size()/2; offset > 0; offset /= 2) 
        {
             localSum += tile32.shfl_down(localSum, offset);
        }
    }

    // write result for this block to global mem
    if (cta.thread_rank() == 0) g_odata[blockIdx.x] = localSum;
}

} // ns linalg_kernels

// end of file
