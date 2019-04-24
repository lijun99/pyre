// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

// my declaration
#include "elementwise.h"
// cuda utitlies
#include "cudalib.h"

// define all cuda kernels 
namespace elementwise_kernels {
    template<typename T>
    __global__ void _fill(T *array, const size_t n, const T value);

    template<typename T1, typename T2>
    __global__ void _conversion(T1 *out, const T2 *in, size_t n);

    template<typename T>
    __global__ void _iadd(T *a1, const T *a2, const size_t n);

    template<typename T>
    __global__ void _isub(T *a1, const T *a2, const size_t n);

    template<typename T>
    __global__ void _imul(T *a1, const T a2, const size_t n);

} // of namespace elementwise_kernels

// fill an array with a given value
template <typename T>
void
cudalib::elementwise::
fill(T *array, const size_t n, const T value, cudaStream_t stream)
{
    dim3 blockSize = NTHREADS;
    dim3 gridSize = IDIVUP(n, NTHREADS);
    elementwise_kernels::_fill<T><<<gridSize, blockSize, 0, stream>>>
        (array, n, value);
    cudaCheckError("cuda error: cudalib::elementwise::set_value");
} 

// explicit specialization
template void cudalib::elementwise::fill<double>(double *, const size_t, const double, cudaStream_t);
template void cudalib::elementwise::fill<float>(float *, const size_t, const float, cudaStream_t);
template void cudalib::elementwise::fill<int>(int *, const size_t, const int, cudaStream_t);
template void cudalib::elementwise::fill<long>(long *, const size_t, const long, cudaStream_t);

// converting data from T2 type to T1 type
template <typename T1, typename T2>
void
cudalib::elementwise::
conversion(T1 *out, const T2 *in, const size_t n, cudaStream_t stream)
{
    dim3 blockSize = NTHREADS;
    dim3 gridSize = IDIVUP(n, NTHREADS);
    elementwise_kernels::_conversion<T1, T2><<<gridSize, blockSize, 0, stream>>>
        (out, in, n);
    cudaCheckError("cuda error: cudalib::elementwise::conversion");
} 

// explicit specialization
template void cudalib::elementwise::conversion<float, double>(float *, const double *, const size_t, cudaStream_t);
template void cudalib::elementwise::conversion<double, float>(double *, const float *, const size_t, cudaStream_t);

// cuda add a1+=a2
template <typename T>
void
cudalib::elementwise::
iadd(T *a1, const T *a2, const size_t n, cudaStream_t stream)
{
    dim3 blockSize = NTHREADS;
    dim3 gridSize = IDIVUP(n, NTHREADS);
    elementwise_kernels::_iadd<T><<<gridSize, blockSize, 0, stream>>>
        (a1, a2, n);
    cudaCheckError("cuda error: cudalib::elementwise::iadd");
} 

// explicit specialization
template void cudalib::elementwise::iadd<float>(float *, const float *, const size_t, cudaStream_t);
template void cudalib::elementwise::iadd<double>(double *, const double *, const size_t, cudaStream_t);
template void cudalib::elementwise::iadd<int>(int *, const int *, const size_t, cudaStream_t);
template void cudalib::elementwise::iadd<long>(long *, const long *, const size_t, cudaStream_t);

// cuda sub a1-=a2

template <typename T>
void
cudalib::elementwise::
isub(T *a1, const T *a2, const size_t n, cudaStream_t stream)
{
    dim3 blockSize = NTHREADS;
    dim3 gridSize = IDIVUP(n, NTHREADS);
    elementwise_kernels::_isub<T><<<gridSize, blockSize, 0, stream>>>
        (a1, a2, n);
    cudaCheckError("cuda error: cudalib::elementwise::isub");
} 

// explicit specialization
template void cudalib::elementwise::isub<float>(float *, const float *, const size_t, cudaStream_t);
template void cudalib::elementwise::isub<double>(double *, const double *, const size_t, cudaStream_t);
template void cudalib::elementwise::isub<int>(int *, const int *, const size_t, cudaStream_t);
template void cudalib::elementwise::isub<long>(long *, const long *, const size_t, cudaStream_t);

// a1*=a2
template <typename T>
void
cudalib::elementwise::
imul(T *a1, const T a2, const size_t n, cudaStream_t stream)
{
    dim3 blockSize = NTHREADS;
    dim3 gridSize = IDIVUP(n, NTHREADS);
    elementwise_kernels::_imul<T><<<gridSize, blockSize, 0, stream>>>
        (a1, a2, n);
    cudaCheckError("cuda error: cudalib::elementwise::imul");
} 

// explicit specialization
template void cudalib::elementwise::imul<float>(float *, const float, const size_t, cudaStream_t);
template void cudalib::elementwise::imul<double>(double *, const double, const size_t, cudaStream_t);
template void cudalib::elementwise::imul<int>(int *, const int, const size_t, cudaStream_t);
template void cudalib::elementwise::imul<long>(long *, const long, const size_t, cudaStream_t);

// cuda kernel for set initial values
template<typename T>
__global__
void
elementwise_kernels::_fill(T *array, const size_t n, const T value)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        array[idx] = value;
}

template<typename T1, typename T2>
__global__
void
elementwise_kernels::_conversion(T1 *out, const T2 *in, size_t n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        out[idx] = (T1)in[idx];
}

template<typename T>
__global__
void
elementwise_kernels::_iadd(T *a1, const T *a2, const size_t n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        a1[idx] += a2[idx];
}

template<typename T>
__global__
void elementwise_kernels::_isub(T *a1, const T *a2, const size_t n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        a1[idx] -= a2[idx];
}

template<typename T>
__global__
void elementwise_kernels::_imul(T *a1, const T a2, const size_t n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        a1[idx] *= a2;
}
// end of file
