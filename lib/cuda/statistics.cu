// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

// my declaration
#include "statistics.h"
// cuda utilities
#include "cudalib.h"
#include "reduction.h"
#include "atomic.h"

// debug
#include <iostream>

#include <cooperative_groups.h>

// alias for cuda cooperative_groups
namespace cg = cooperative_groups;

// declare cuda kernels at first
namespace statistics_kernels {

    /* current cuda (nvcc) limitations with template classes and cuda kernels
      1. using extern __shared__ T sdata[] isn't currently supported by nvcc for multiple instantiations
         therefore, we allocate shared memory in kernels and need a blockSize template parameter
      2. cudaLaunchKernel doesn't support a template kernel
         therefore, the launching form <T><<< ... >>> has to be use
    */

    template<typename T, const int blockSize>
    __global__ void _sum(T * const sum, const T* const gdata, const size_t n, const size_t stride);

    template<typename T, const int blockSize>
    __global__ void _sumsq(T * const sum,  const T* const gdata, const T * const mean, const size_t n, const size_t stride);

    template<typename T, const int blockSize>
    __global__ void _max(T * const max, const T* const gdata, const size_t n, const size_t stride);

    template<typename T, const int blockSize>
    __global__ void _min(T * const min, const T* const gdata, const size_t n, const size_t stride);

    template<typename T, const int blockSize>
    __device__ void _mean_oneblock(T* const mean, const T * const gdata, const size_t n, const size_t stride);

    template<typename T, const int blockSize>
    __device__ void _std_oneblock(T* std, const T * const gdata,  const T * const mean,
        const size_t n, const size_t stride, const int ddof);

    template<typename T, const int blockSize>
    __global__ void _matrix_mean_over_rows(const cuda_matrix gmatrix, cuda_vector gmean);

    template<typename T, const int blockSize>
    __global__ void _matrix_mean_over_cols(const cuda_matrix gmatrix, cuda_vector gmean);

    template<typename T, const int blockSize>
    __global__ void _matrix_mean_std_over_rows(const cuda_matrix gmatrix, cuda_vector gmean,
        cuda_vector gstd, const int ddof);

    template<typename T, const int blockSize>
    __global__ void _matrix_mean_std_over_cols(const cuda_matrix gmatrix, cuda_vector gmean,
        cuda_vector gstd, const int ddof);

    template<typename T, const int blockSize>
    __global__ void _l1norm(T * const norm, const T* const gdata, const size_t n, const size_t stride);

    template<typename T, const int blockSize>
    __global__ void _l2norm(T * const norm, const T* const gdata, const size_t n, const size_t stride);

    template<typename T, const int blockSize>
    __global__ void _linfnorm(T * const norm, const T* const gdata, const size_t n, const size_t stride);

    template<typename T, const int blockSize>
    __global__ void _covariance(T * const cov, const T* const v1, const T* const v1mean,
        const T * const v2, const T* const v2mean, const size_t n, const size_t stride);

    template<typename T, const int blockSize>
    __device__ void _covariance_oneblock(T * const cov, const T * const x, const T * const xmean,
        const T * const y, const T* const ymean, const size_t n, const size_t stride, const int ddof);

    template<typename T, const int blockSize>
    __global__ void _covariance_batched(T * const cov, const T* const x, const T* const y, const size_t n,
        const size_t batch_stride, const size_t elem_stride, const int ddof);

    template<typename T, const int blockSize>
    __global__ void _correlation_batched(T * const cor, const T* const x, const T* const y, const size_t n,
        const size_t batch_stride, const size_t elem_stride);

}

// put specialization inside namespace for gcc<7 compatibility
namespace cudalib {
    namespace statistics {
// sum
template <typename T>
T sum(const T* const gdata, const size_t n, const size_t stride,  cudaStream_t stream)
{
    // work data
    T * gsum, sum;
    cudaSafeCall(cudaMalloc((void **)&gsum, sizeof(T)));
    // set initial value of (global) sum to 0
    cudaSafeCall(cudaMemset(gsum, 0, sizeof(T)));

    // kernel launch parameters
    const int blockSize = BLOCKDIM;
    const int gridSize = IDIVUP(n, blockSize);

    // call cuda kernel
    statistics_kernels::_sum<T, blockSize><<<gridSize, blockSize, 0, stream>>>
        (gsum, gdata, n, stride);
    cudaCheckError("statistics_kernels::_sum");

    // copy result to cpu
    cudaSafeCall(cudaMemcpyAsync(&sum, gsum, sizeof(T), cudaMemcpyDefault, stream));
    // free work data
    cudaSafeCall(cudaFree(gsum));
    // all done
    return sum;
}

// explicit instantiation
template float sum<float> (const float * const, const size_t, const size_t, cudaStream_t);
template double sum<double> (const double * const, const size_t, const size_t, cudaStream_t);
template int sum<int> (const int * const, const size_t, const size_t, cudaStream_t);
template unsigned long long sum<unsigned long long> (const unsigned long long * const, const size_t, const size_t, cudaStream_t);

// max
template <typename T>
T max(const T* const gdata, const size_t n, const size_t stride, cudaStream_t stream)
{
    // work data
    T * gmax, max;
    cudaSafeCall(cudaMalloc((void **)&gmax, sizeof(T)));
    // initialize the gmax value to first data
    cudaSafeCall(cudaMemcpyAsync(gmax, gdata, sizeof(T), cudaMemcpyDefault, stream));

    // kernel launch parameters
    const int blockSize = BLOCKDIM;
    const int gridSize = IDIVUP(n, blockSize);

    // call cuda kernel
    statistics_kernels::_max<T, blockSize><<<gridSize, blockSize, 0, stream>>>
        (gmax, gdata, n, stride);
    cudaCheckError("statistics_kernels::_max");

    // copy result to cpu
    cudaSafeCall(cudaMemcpyAsync(&max, gmax, sizeof(T), cudaMemcpyDefault, stream));
    // free work data
    cudaSafeCall(cudaFree(gmax));
    // all done
    return max;
}

// explicit instantiation
template float max<float> (const float * const, const size_t, const size_t, cudaStream_t);
template double max<double> (const double * const, const size_t, const size_t, cudaStream_t);
template int max<int> (const int * const, const size_t, const size_t, cudaStream_t);

// min
template <typename T>
T min(const T* const gdata, const size_t n, const size_t stride, cudaStream_t stream)
{
    // work data
    T * gmin, min;
    cudaSafeCall(cudaMalloc((void **)&gmin, sizeof(T)));
    // initialize the gmax value to first data
    cudaSafeCall(cudaMemcpyAsync(gmin, gdata, sizeof(T), cudaMemcpyDefault, stream));

    // kernel launch parameters
    const int blockSize = BLOCKDIM;
    const int gridSize = IDIVUP(n, blockSize);

    // call cuda kernel
    statistics_kernels::_min<T, blockSize><<<gridSize, blockSize, 0, stream>>>
        (gmin, gdata, n, stride);
    cudaCheckError("statistics_kernels::_min");

    // copy result to cpu
    cudaSafeCall(cudaMemcpyAsync(&min, gmin, sizeof(T), cudaMemcpyDefault, stream));
    // free work data
    cudaSafeCall(cudaFree(gmin));
    // all done
    return min;
}

// explicit instantiation
template float min<float> (const float * const, const size_t, const size_t, cudaStream_t);
template double min<double> (const double * const, const size_t, const size_t, cudaStream_t);
template int min<int> (const int * const, const size_t, const size_t, cudaStream_t);

// mean
template <typename T>
T mean(const T* const gdata, const size_t n, const size_t stride, cudaStream_t stream)
{
    // call sum and divided by n
    return sum<T>(gdata, n, stride, stream)/n;
}

// explicit instantiation
template float mean<float> (const float * const, const size_t, const size_t, cudaStream_t);
template double mean<double> (const double * const, const size_t, const size_t, cudaStream_t);

// standard deviation
template <typename T>
T std(const T* const gdata, const T mean, const size_t n, const size_t stride, const int ddof, cudaStream_t stream)
{
    // work data
    T * gsum, * gmean, std;
    cudaSafeCall(cudaMalloc((void **)&gsum, sizeof(T)));
    cudaSafeCall(cudaMemset(gsum, 0, sizeof(T)));
    cudaSafeCall(cudaMalloc((void **)&gmean, sizeof(T)));
    cudaSafeCall(cudaMemcpyAsync(gmean, &mean, sizeof(T), cudaMemcpyDefault, stream));

    // kernel launch parameters
    const int blockSize = BLOCKDIM;
    const int gridSize = IDIVUP(n, blockSize);

    // call cuda kernel
    statistics_kernels::_sumsq<T, blockSize><<<gridSize, blockSize, 0, stream>>>
        (gsum, gdata, (const T*)gmean, n, stride);
    cudaCheckError("statistics_kernels::_sumsq");

    // copy result to cpu
    cudaSafeCall(cudaMemcpyAsync(&std, gsum, sizeof(T), cudaMemcpyDefault, stream));
    std = sqrt(std/(n-ddof));
    // free work data
    cudaSafeCall(cudaFree(gsum));
    cudaSafeCall(cudaFree(gmean));
    // all done
    return std;
}

// explicit instantiation
template float std<float>(const float* const, const float, const size_t, const size_t, const int, cudaStream_t);
template double std<double>(const double* const, const double, const size_t, const size_t, const int, cudaStream_t);

// average over rows, results are vectors with size cols (size2)
template <typename T>
void matrix_mean_over_rows(const cuda_matrix gmatrix, cuda_vector mean, cudaStream_t stream)
{
    // set mean to 0 at first
    cudaSafeCall(cudaMemset(mean.data, 0, mean.nbytes));

    // kernel launch parameters
    const int blockSize = MAXTHREADS;
    const int gridSize = gmatrix.size2;

    // call cuda kernel
    statistics_kernels::_matrix_mean_over_rows<T, blockSize><<<gridSize, blockSize, 0, stream>>>
            (gmatrix, mean);
    cudaCheckError("statistics_kernels::_matrix_mean_over_rows");
    // all done
}

// explicit instantiation
template void matrix_mean_over_rows<float>(const cuda_matrix, cuda_vector, cudaStream_t);
template void matrix_mean_over_rows<double>(const cuda_matrix, cuda_vector, cudaStream_t);

// average over cols, results are vectors with size rows (size1)
template <typename T>
void matrix_mean_over_cols(const cuda_matrix gmatrix, cuda_vector mean, cudaStream_t stream)
{
    // set mean to 0 at first
    cudaSafeCall(cudaMemset(mean.data, 0, mean.nbytes));

    // kernel launch parameters
    const int blockSize = MAXTHREADS;
    const int gridSize = gmatrix.size1;

    // call cuda kernel
    statistics_kernels::_matrix_mean_over_cols<T, blockSize><<<gridSize, blockSize, 0, stream>>>
            (gmatrix, mean);
    cudaCheckError("statistics_kernels::_matrix_mean_over_cols");
    // all done
}

// explicit instantiation
template void matrix_mean_over_cols<float>(const cuda_matrix, cuda_vector, cudaStream_t);
template void matrix_mean_over_cols<double>(const cuda_matrix, cuda_vector, cudaStream_t);


// average over rows, results are vectors with size cols (size2)
template <typename T>
void matrix_mean_std_over_rows(const cuda_matrix gmatrix, cuda_vector mean, cuda_vector std, const int ddof,
    cudaStream_t stream)
{
    // set mean to 0 at first
    cudaSafeCall(cudaMemset(mean.data, 0, mean.nbytes));

    // kernel launch parameters
    const int blockSize = MAXTHREADS;
    const int gridSize = gmatrix.size2;

    // call cuda kernel
    statistics_kernels::_matrix_mean_std_over_rows<T, blockSize><<<gridSize, blockSize, 0, stream>>>
            (gmatrix, mean, std, ddof);
    cudaCheckError("statistics_kernels::_matrix_mean_std_over_rows");
    // all done
}

// explicit instantiation
template void matrix_mean_std_over_rows<float>(const cuda_matrix, cuda_vector, cuda_vector, const int, cudaStream_t);
template void matrix_mean_std_over_rows<double>(const cuda_matrix, cuda_vector, cuda_vector, const int, cudaStream_t);

// average over cols, results are vectors with size rows (size1)
template <typename T>
void matrix_mean_std_over_cols(const cuda_matrix gmatrix, cuda_vector mean, cuda_vector std,
    const int ddof, cudaStream_t stream)
{
    // set mean to 0 at first
    cudaSafeCall(cudaMemset(mean.data, 0, mean.nbytes));

    // kernel launch parameters
    const int blockSize = MAXTHREADS;
    const int gridSize = gmatrix.size1;

    // call cuda kernel
    statistics_kernels::_matrix_mean_std_over_cols<T, blockSize><<<gridSize, blockSize, 0, stream>>>
            (gmatrix, mean, std, ddof);
    cudaCheckError("statistics_kernels::_matrix_mean_std_over_cols");
    // all done
}

// explicit instantiation
template void matrix_mean_std_over_cols<float>(const cuda_matrix, cuda_vector, cuda_vector, const int, cudaStream_t);
template void matrix_mean_std_over_cols<double>(const cuda_matrix, cuda_vector, cuda_vector, const int, cudaStream_t);

template <typename T>
T L1norm(const T* const gdata, const size_t n, const size_t stride, cudaStream_t stream)
{
    // work data
    T * gnorm, norm;
    cudaSafeCall(cudaMalloc((void **)&gnorm, sizeof(T)));
    cudaSafeCall(cudaMemset(gnorm, 0, sizeof(T)));

    // kernel launch parameters
    const int blockSize = BLOCKDIM;
    const int gridSize = IDIVUP(n, blockSize);

    // call cuda mean_kernel
    statistics_kernels::_l1norm<T, blockSize><<<gridSize, blockSize, 0, stream>>>
        (gnorm, gdata, n, stride);
    cudaCheckError("statistics_kernels::_l1norm");

    // copy result to cpu
    cudaSafeCall(cudaMemcpyAsync(&norm, gnorm, sizeof(T), cudaMemcpyDefault, stream));
    // free work data
    cudaSafeCall(cudaFree(gnorm));
    // all done
    return norm;
}

// explicit instantiation
template float L1norm <float>(const float* const, const size_t, const size_t, cudaStream_t);
template double L1norm <double>(const double* const, const size_t, const size_t, cudaStream_t);

template <typename T>
T L2norm(const T* const gdata, const size_t n, const size_t stride, cudaStream_t stream)
{
    // work data
    T * gnorm, norm;
    cudaSafeCall(cudaMalloc((void **)&gnorm, sizeof(T)));
    cudaSafeCall(cudaMemset(gnorm, 0, sizeof(T)));

    // kernel launch parameters
    const int blockSize = BLOCKDIM;
    const int gridSize = IDIVUP(n, blockSize);

    // call cuda mean_kernel
    statistics_kernels::_l2norm<T, blockSize><<<gridSize, blockSize, 0, stream>>>
            (gnorm, gdata, n, stride);
    cudaCheckError("statistics_kernels::_l2norm");

    // copy result to cpu
    cudaSafeCall(cudaMemcpyAsync(&norm, gnorm, sizeof(T), cudaMemcpyDefault, stream));
    // free work data
    cudaSafeCall(cudaFree(gnorm));
    // all done
    return norm;
}

// explicit instantiation
template float L2norm <float>(const float* const, const size_t, const size_t, cudaStream_t);
template double L2norm <double>(const double* const, const size_t, const size_t, cudaStream_t);

template <typename T>
T Linfnorm(const T* const gdata, const size_t n, const size_t stride, cudaStream_t stream)
{
    // work data
    T * gnorm, norm;
    cudaSafeCall(cudaMalloc((void **)&gnorm, sizeof(T)));
    cudaSafeCall(cudaMemset(gnorm, 0, sizeof(T)));

    // kernel launch parameters
    const int blockSize = BLOCKDIM;
    const int gridSize = IDIVUP(n, blockSize);

    // call cuda kernel
    statistics_kernels::_linfnorm<T, blockSize> <<<gridSize, blockSize, 0, stream>>>
        (gnorm, gdata, n, stride);
    cudaCheckError("statistics_kernels::_linfnorm");

    // copy result to cpu
    cudaSafeCall(cudaMemcpyAsync(&norm, gnorm, sizeof(T), cudaMemcpyDefault, stream));
    // free work data
    cudaSafeCall(cudaFree(gnorm));
    // all done
    return norm;
}

// explicit instantiation
template float Linfnorm <float>(const float* const, const size_t, const size_t, cudaStream_t);
template double Linfnorm <double>(const double* const, const size_t, const size_t, cudaStream_t);


// covariance cov(x,y) = E[(x-E[x])(y-E[y])] = E(xy) -E(x)E(y)
// with mean as input
template <typename T>
T covariance_wmean(const T* const x, const T xmean, const T* const y, const T ymean,
    const size_t n, const size_t stride, const int ddof, cudaStream_t stream)
{
    // work data
    T * gcov, *gxmean, *gymean;
    cudaSafeCall(cudaMalloc((void **)&gcov, sizeof(T)));
    cudaSafeCall(cudaMemset(gcov, 0, sizeof(T)));

    cudaSafeCall(cudaMalloc((void **)&gxmean, sizeof(T)));
    cudaSafeCall(cudaMemcpyAsync(gxmean, &xmean, sizeof(T), cudaMemcpyDefault, stream));
    cudaSafeCall(cudaMalloc((void **)&gymean, sizeof(T)));
    cudaSafeCall(cudaMemcpyAsync(gymean, &ymean, sizeof(T), cudaMemcpyDefault, stream));

    // kernel launch parameters
    const int blockSize = BLOCKDIM;
    const int gridSize = IDIVUP(n, blockSize);

    // call cuda kernel
    statistics_kernels::_covariance<T, blockSize> <<<gridSize, blockSize, 0, stream>>>
        (gcov, x, gxmean, y, gymean, n, stride);
    cudaCheckError("statistics_kernels::_covariance");

    // copy result to cpu
    T cov;
    cudaSafeCall(cudaMemcpyAsync(&cov, gcov, sizeof(T), cudaMemcpyDefault, stream));

    // divided by n-ddof (delta degrees of freedom)
    cov = cov/(n-ddof);

    // free work data
    cudaSafeCall(cudaFree(gcov));
    cudaSafeCall(cudaFree(gxmean));
    cudaSafeCall(cudaFree(gymean));

    // all done
    return cov;
}

// explicit instantiation
template float covariance_wmean<float>(const float * const, const float, const float * const, const float,
    const size_t, const size_t, const int, cudaStream_t);
template double covariance_wmean<double>(const double * const, const double, const double * const, const double,
    const size_t, const size_t, const int, cudaStream_t);

// without mean as input
template <typename T>
T covariance(const T* const x, const T* const y,
    const size_t n, const size_t stride, const int ddof, cudaStream_t stream)
{
    // caluclate mean values of both vectors
    T xmean = mean<T>(x, n, stride, stream);
    T ymean = mean<T>(y, n, stride, stream);

    // call covariance with mean values
    return covariance_wmean<T>(x, xmean, y, ymean, n, stride, ddof, stream);
}

// explicit instantiation
template float covariance<float>(const float * const, const float * const,
    const size_t, const size_t, const int, cudaStream_t);
template double covariance<double>(const double * const, const double * const,
    const size_t, const size_t, const int, cudaStream_t);

// correlation cov(x,y) / (std(x) std(y))
template <typename T>
T correlation(const T* const x, const T* const y, const size_t n, const size_t stride, cudaStream_t stream)
{
    // compute mean
    T xmean = mean<T>(x, n, stride, stream);
    T ymean = mean<T>(y, n, stride, stream);

    // (n-ddof) factor will cancel out
    const int ddof = 1;
    // compute covariance
    T cov = covariance_wmean<T>(x, xmean, y, ymean, n, stride, ddof, stream);
    // compute std
    T xstd = std<T>(x, xmean, n, stride, ddof, stream);
    T ystd = std<T>(y, ymean, n, stride, ddof, stream);

    // return correlation
    return cov/(xstd*ystd);
}

// explicit instantiation
template float correlation <float>(const float * const, const float * const, const size_t, const size_t, cudaStream_t);
template double correlation <double>(const double * const, const double * const, const size_t, const size_t, cudaStream_t);

// covariance between a batch of vectors
template <typename T>
void covariance_batched(T* const cor, const T* const x, const T* const y, const size_t n,
    const size_t batch, const size_t batch_stride, const size_t elem_stride, const int ddof, cudaStream_t stream)
{
    // kernel launch parameters
    const int blockSize = MAXTHREADS;
    const int gridSize = batch;

    // call cuda kernel
    statistics_kernels::_covariance_batched<T, blockSize><<<gridSize, blockSize, 0, stream>>>
            (cor, x, y, n, batch_stride, elem_stride, ddof);
    cudaCheckError("statistics_kernels::_covariance_batched");
    // all done

}

// explicit instantiation
template void covariance_batched<float>(float * const, const float * const, const float * const,
    const size_t, const size_t, const size_t, const size_t, const int, cudaStream_t);
template void covariance_batched<double>(double * const, const double * const, const double * const,
    const size_t, const size_t, const size_t, const size_t, const int, cudaStream_t);


// covariance for a batch of vectors
template <typename T>
void correlation_batched(T* const cor, const T* const x, const T* const y, const size_t n,
    const size_t batch, const size_t batch_stride, const size_t elem_stride, cudaStream_t stream)
{
    // kernel launch parameters
    const int blockSize = MAXTHREADS;
    const int gridSize = batch;

    // call cuda kernel
    statistics_kernels::_correlation_batched<T, blockSize><<<gridSize, blockSize, 0, stream>>>
            (cor, x, y, n, batch_stride, elem_stride);
    cudaCheckError("statistics_kernels::_correlation_batched");
    // all done

}

// explicit instantiation
template void correlation_batched<float>(float * const, const float * const, const float * const,
    const size_t, const size_t, const size_t, const size_t, cudaStream_t);
template void correlation_batched<double>(double * const, const double * const, const double * const,
    const size_t, const size_t, const size_t, const size_t, cudaStream_t);

    } // of namespace statistics
} // of namespace cudalib


//************************* kernels ****************************
namespace statistics_kernels {

// sum reduction with multiple thread blocks
// good for one large vector
template<typename T, const int blockSize>
__global__ void _sum(T * const sum, const T* const gdata, const size_t n, const size_t stride)
{
    // shared memory with blockSize
    __shared__ T sdata[blockSize];
    // get global thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // get local thread id of current block
    auto cta = cg::this_thread_block();
    int rank = cta.thread_rank();

    // copy data to shared memory
    T local_sum = (tid < n) ? gdata[tid*stride] : 0;
    sdata[rank] = local_sum;
    cg::sync(cta);

    // call block sum
    local_sum = reduction_kernels::sum_reduce_block<T>(sdata, cta);

    // sum over different blocks
    // only for first thread of each block
    if (rank == 0)
        ::atomicAdd(sum, local_sum);
    // all done
}

// sum reduction with one thread block
// good for a batch of vectors
// each block sums over one vector
// defined as device function, to be called by another __global__ function
template<typename T, const int blockSize>
__device__ void _sum_oneblock(T * const sum, const T* const gdata, const size_t n, const size_t stride)
{
    // shared memory with blockSize
    __shared__ T sdata[blockSize];
    // get the block thread group
    auto cta = cg::this_thread_block();
    // thread id iterate over rows
    int rank = cta.thread_rank();

    // each thread sums rows with interval blockSize
    T local_sum = 0;
    for(int i=rank; i<n; i+=blockSize)
        local_sum += gdata[i*stride];

    sdata[rank] = local_sum;
    cg::sync(cta);
    // call block sum
    local_sum = reduction_kernels::sum_reduce_block<T>(sdata, cta);

    if (rank == 0) {
        sum[0] = local_sum;
    }
}


template<typename T, const int blockSize>
__global__ void _sumsq(T * const sum,  const T* const gdata, const T * const mean, const size_t n, const size_t stride)
{
    // shared memory with blockSize
    __shared__ T sdata[blockSize];
    // get global thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // get current block and local thread id
    auto cta = cg::this_thread_block();
    int rank = cta.thread_rank();

    // compute local (gdata[i]-mean)**2
    T local_sum = 0;
    if (tid<n) {
        T diff = gdata[tid*stride]-mean[0];
        local_sum = diff*diff;
    }
    // copy to shared memory for reduction
    sdata[rank] = local_sum;
    cg::sync(cta);
    // call block sum
    local_sum = reduction_kernels::sum_reduce_block<T>(sdata, cta);
    // sum over all blocks
    if (rank == 0) ::atomicAdd(sum, local_sum);
}

template<typename T, const int blockSize>
__global__ void _max(T * const max, const T* const gdata, const size_t n, const size_t stride)
{
    // shared memory with blockSize
    __shared__ T sdata[blockSize];
    // get thread id
    auto cta = cg::this_thread_block();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //
    T local_max = (tid < n) ? gdata[tid*stride] : gdata[0];

    sdata[cta.thread_rank()] = local_max;
    cg::sync(cta);
    // call block max
    local_max = reduction_kernels::max_reduce_block<T>(sdata, cta);

    if (cta.thread_rank() == 0) {
        ::atomicMax(max, local_max);
    }
}

template<typename T, const int blockSize>
__global__ void _min(T * const min, const T* const gdata, const size_t n, const size_t stride)
{
    // shared memory with blockSize
    __shared__ T sdata[blockSize];
    // get thread id
    auto cta = cg::this_thread_block();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // copy values to shared memory
    T local_min = (tid < n) ? gdata[tid*stride] : gdata[0];
    sdata[cta.thread_rank()] = local_min;
    cg::sync(cta);

    // call block min (only tid=0 return the correct value)
    local_min = reduction_kernels::min_reduce_block<T>(sdata, cta);

    // for each block, use atomic function to update the global value
    if(cta.thread_rank() == 0) {
        ::atomicMin(min, local_min);
    }
}


template<typename T, const int blockSize>
__device__ void _mean_oneblock(T * mean, const T * gdata, const size_t n, const size_t stride)
{
    // shared memory with blockSize
    __shared__ T sdata[blockSize];
    // get the block thread group
    auto cta = cg::this_thread_block();
    // thread id iterate over rows
    // note tid = threadIdx.x, not blockIdx.x*blockDim.x + threadIdx.x
    int tid = cta.thread_rank();

    // each thread sums rows with interval blockSize
    T local_sum = 0;
    for(int i=tid; i<n; i+=blockDim.x)
        local_sum += gdata[i*stride];

    sdata[tid] = local_sum;
    cg::sync(cta);
    // call block sum
    local_sum = reduction_kernels::sum_reduce_block<T>(sdata, cta);

    if (tid == 0) {
        mean[0] = local_sum/n;
    }
}

template<typename T, const int blockSize>
__device__ void _std_oneblock(T* const std, const T* const gdata, const T * const mean,
    const size_t n, const size_t stride, const int ddof)
{
    // shared memory with blockSize
    __shared__ T sdata[blockSize];
    // get the block thread group
    auto cta = cg::this_thread_block();
    // thread id iterate over rows
    int tid = cta.thread_rank();

    // each thread sums rows with interval blockSize
    T local_sum = 0;
    for(int i=tid; i<n; i+=blockDim.x) {
        T diff = gdata[i*stride] - mean[0];
        local_sum += diff*diff;
    }
    // copy to shared memory
    sdata[tid] = local_sum;
    cg::sync(cta);
    // call block sum
    local_sum = reduction_kernels::sum_reduce_block<T>(sdata, cta);

    if (tid == 0) {
        std[0] = sqrt(local_sum/(n-ddof));
    }
}

template<typename T, const int blockSize>
__global__ void _matrix_mean_over_rows(const cuda_matrix gmatrix, cuda_vector gmean)
{
    // block id as column index
    int col = blockIdx.x;
    // starting pointer for this column
    const T * gdata = (const T *)gmatrix.data + col;
    T * mean = (T *)gmean.data + col;
    // calculate mean
    _mean_oneblock<T, blockSize>(mean, gdata, gmatrix.size1, gmatrix.size2);
}

template<typename T, const int blockSize>
__global__ void _matrix_mean_over_cols(const cuda_matrix gmatrix, cuda_vector gmean)
{
    // block id as row index
    int row = blockIdx.x;
    // starting pointer for this row
    const T * gdata = (const T *)gmatrix.data + row*gmatrix.size2;
    T * mean = (T *)gmean.data + row;
    // calculate mean
    _mean_oneblock<T, blockSize>(mean, gdata, gmatrix.size2, 1);
}

template<typename T, const int blockSize>
__global__ void _matrix_mean_std_over_rows(const cuda_matrix gmatrix, cuda_vector gmean, cuda_vector gstd,
    const int ddof)
{
    // block id as column index
    int col = blockIdx.x;
    // starting pointer for this column
    const T * gdata = (const T *)gmatrix.data + col;
    T * mean = (T *)gmean.data + col;
    T * std = (T *)gstd.data + col;

    // calculate mean
    _mean_oneblock<T, blockSize>(mean, gdata, gmatrix.size1, gmatrix.size2);
    __syncthreads();
    // calculate std
    _std_oneblock<T, blockSize>(std, gdata, mean, gmatrix.size1, gmatrix.size2, ddof);
}

template<typename T, const int blockSize>
__global__ void _matrix_mean_std_over_cols(const cuda_matrix gmatrix, cuda_vector gmean, cuda_vector gstd,
    const int ddof)
{
    // block id as row index
    int row = blockIdx.x;
    // starting pointer for this row
    const T * gdata = (const T *)gmatrix.data + row*gmatrix.size2;
    T * mean = (T *)gmean.data + row;
    T * std = (T *)gstd.data + row;

    // calculate mean
    _mean_oneblock<T, blockSize>(mean, gdata, gmatrix.size2, 1);
    __syncthreads();
    // calculate std
    _std_oneblock<T, blockSize>(std, gdata, mean, gmatrix.size2, 1, ddof);
}


template<typename T, const int blockSize>
__global__ void _l1norm(T * const norm, const T* const gdata, const size_t n, const size_t stride)
{
    // shared memory with blockSize
    __shared__ T sdata[blockSize];
    // get thread id
    auto cta = cg::this_thread_block();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //
    T local_sum = (tid < n) ? abs(gdata[tid*stride]): 0;

    sdata[cta.thread_rank()] = local_sum;
    cg::sync(cta);
    // call block sum
    local_sum = reduction_kernels::sum_reduce_block<T>(sdata, cta);

    if (cta.thread_rank() == 0) {
        ::atomicAdd(norm, local_sum);
    }
}

template<typename T, const int blockSize>
__global__ void _l2norm(T * const norm, const T* const gdata, const size_t n, const size_t stride)
{
    // shared memory with blockSize
    __shared__ T sdata[blockSize];
    // get thread id
    auto cta = cg::this_thread_block();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //
    T local_sum = 0;
    if (tid < n) {
        auto value = gdata[tid*stride];
        local_sum += value*value;
    }

    sdata[cta.thread_rank()] = local_sum;
    cg::sync(cta);
    // call block sum
    local_sum = reduction_kernels::sum_reduce_block<T>(sdata, cta);

    if (cta.thread_rank() == 0) {
        ::atomicAdd(norm, local_sum);
    }
}

template<typename T, const int blockSize>
__global__ void _linfnorm(T * const norm, const T* const gdata, const size_t n, const size_t stride)
{
    // shared memory with blockSize
    __shared__ T sdata[blockSize];
    // get thread id
    auto cta = cg::this_thread_block();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //
    T local_max = (tid < n) ? abs(gdata[tid*stride]): 0;

    sdata[cta.thread_rank()] = local_max;
    cg::sync(cta);
    // call block max
    local_max = reduction_kernels::max_reduce_block<T>(sdata, cta);

    if (cta.thread_rank() == 0) {
        ::atomicMax(norm, local_max);
    }
}

template<typename T, const int blockSize>
__global__ void _covariance(T * const cov, const T* const v1, const T * const v1mean,
    const T * const v2, const T * const v2mean, const size_t n, const size_t stride)
{
    // shared memory with blockSize
    __shared__ T sdata[blockSize];
    // get thread id
    auto cta = cg::this_thread_block();
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    //
    T local_sum = (tid < n) ?
        (v1[tid*stride]-v1mean[0])*(v2[tid*stride]-v2mean[0]) : 0 ;

    sdata[cta.thread_rank()] = local_sum;
    cg::sync(cta);
    // call block sum
    local_sum = reduction_kernels::sum_reduce_block<T>(sdata, cta);

    if (cta.thread_rank() == 0) {
        ::atomicAdd(cov, local_sum);
    }
}

template<typename T, const int blockSize>
__device__ void _covariance_oneblock(T * const cov, const T * const x, const T * const xmean,
    const T * const y, const T* const ymean, const size_t n, const size_t stride, const int ddof)
{
    // shared memory with blockSize
    __shared__ T sdata[blockSize];
    // get the block thread group
    auto cta = cg::this_thread_block();
    // thread id iterate over rows
    int tid = cta.thread_rank();

    // each thread sums rows with interval blockSize
    T local_sum = 0;
    for(int i=tid; i<n; i+=blockDim.x) {
        local_sum += (x[i*stride] - xmean[0])*(y[i*stride]-ymean[0]);
    }

    // copy to shared memory
    sdata[tid] = local_sum;
    cg::sync(cta);
    // call block sum
    local_sum = reduction_kernels::sum_reduce_block<T>(sdata, cta);
    // set the final result
    if (tid == 0) {
        cov[0] = local_sum/(n-ddof);
    }
}

// covariance for a batch of vectors
template<typename T, const int blockSize>
__global__ void _covariance_batched(T * const cov, const T* const x, const T* const y, const size_t n,
    const size_t batch_stride, const size_t elem_stride, const int ddof)
{
    // block id as batch index
    int batch = blockIdx.x;
    // starting pointer for this row
    const T * xv = x + batch*batch_stride;
    const T * yv = y + batch*batch_stride;
    T * covv = cov + batch;

    // get the block thread group
    auto cta = cg::this_thread_block();

    // calculate mean and std for x vector
    _mean_oneblock<T, blockSize>(covv, xv, n, elem_stride);
    cg::sync(cta);
    T xmean = covv[0];

    // calculate mean and std for y vector
    _mean_oneblock<T, blockSize>(covv, yv, n, elem_stride);
    cg::sync(cta);
    T ymean = covv[0];

    // calculate \sum(x-xmean)(y-ymean)
    _covariance_oneblock<T, blockSize>(covv, xv, &xmean, yv, &ymean, n, elem_stride, ddof);

}

// correlation for a batch of vectors
template<typename T, const int blockSize>
__global__ void _correlation_batched(T * const cor, const T* const x, const T* const y, const size_t n,
    const size_t batch_stride, const size_t elem_stride)
{
    // block id as batch index
    int batch = blockIdx.x;
    // starting pointer for this row
    const T * xv = x + batch*batch_stride;
    const T * yv = y + batch*batch_stride;
    T * corv = cor + batch;

    // delta degrees of freedom n-ddof, will cancel out
    const int ddof = 1;

    // get the thread block group
    auto cta = cg::this_thread_block();
    // calculate mean and std for x vector
    _mean_oneblock<T, blockSize>(corv, xv, n, elem_stride);
    cg::sync(cta);
    T xmean = corv[0];
    _std_oneblock<T, blockSize>(corv, xv, &xmean, n, elem_stride, ddof);
    cg::sync(cta);
    T xstd = corv[0];
    // calculate mean and std for y vector
    _mean_oneblock<T, blockSize>(corv, yv, n, elem_stride);
    cg::sync(cta);
    T ymean = corv[0];
    _std_oneblock<T, blockSize>(corv, yv, &ymean, n, elem_stride, ddof);
    cg::sync(cta);
    T ystd = corv[0];

    // calculate \sum(x-xmean)(y-ymean)
    _covariance_oneblock<T, blockSize>(corv, xv, &xmean, yv, &ymean, n, elem_stride, ddof);
    cg::sync(cta);

    if(cta.thread_rank() == 0) {
        corv[0] /= xstd*ystd;
    }

}


} // of namespace statistics_kernels
// ********************** end of kernels **********************

// end of file
