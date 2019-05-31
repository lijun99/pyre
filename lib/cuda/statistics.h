// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

/// @file statistics.h
/// @brief statistics functions within a block

#ifndef cudalib_statistics_h
#define cudalib_statistics_h

#include "cuda_matrix.h"
#include "cuda_vector.h"

namespace cudalib {
    namespace statistics {

        // sum, mean, std (1/(N-1)), max/min
        template <typename T>
        T sum(const T* const gdata, const size_t n, const size_t stride=1, cudaStream_t stream=0);

        template <typename T>
        T max(const T* const gdata, const size_t n, const size_t stride=1, cudaStream_t stream=0);

        template <typename T>
        T min(const T* const gdata, const size_t n, const size_t stride=1, cudaStream_t stream=0);

        template <typename T>
        T mean(const T* const gdata, const size_t n, const size_t stride=1, cudaStream_t stream=0);

        template <typename T>
        T std(const T* const gdata, const T mean, const size_t n, const size_t stride=1,
            const int ddof =1, cudaStream_t stream=0);

        // matrix sum and std over row or cols
        // use cuda_matrix/vector struct as inputs
        template <typename T>
        void matrix_mean_over_rows(const cuda_matrix m, cuda_vector mean, cudaStream_t stream=0);

        template <typename T>
        void matrix_mean_over_cols(const cuda_matrix m, cuda_vector mean, cudaStream_t stream=0);

        template <typename T>
        void matrix_mean_std_over_rows(const cuda_matrix m, cuda_vector mean, cuda_vector std,
            const int ddof=1, cudaStream_t stream=0);

        template <typename T>
        void matrix_mean_std_over_cols(const cuda_matrix m, cuda_vector mean, cuda_vector std,
            const int ddof=1, cudaStream_t stream=0);

        // norms - L1, L2, L-Infinity for a vector
        template <typename T>
        T L1norm(const T* const gdata, const size_t n, const size_t stride=1, cudaStream_t stream=0);

        template <typename T>
        T L2norm(const T* const gdata, const size_t n, const size_t stride=1, cudaStream_t stream=0);

        template <typename T>
        T Linfnorm(const T* const gdata, const size_t n, const size_t stride=1, cudaStream_t stream=0);

        // covariance for vector
        template <typename T>
        T covariance(const T* const v1, const T* const v2, const size_t n, const size_t stride=1,
            const int ddof=1, cudaStream_t stream=0);

        // covariance for vector
        template <typename T>
        T correlation(const T* const v1, const T* const v2, const size_t n, const size_t stride=1,
            cudaStream_t stream=0);

        // covariance for a batch of vectors (e.g. matrix along row or col)
        template <typename T>
        void covariance_batched(T * const cov, const T* const v1, const T* const v2,
            const size_t n, const size_t batch,
            const size_t batch_stride, const size_t elem_stride,
            const int ddof=1,
            cudaStream_t stream=0);

        // correlation for a batch of vectors (e.g. matrix along row or col)
        template <typename T>
        void correlation_batched(T* const cor, const T* const x, const T* const y,
            const size_t n, const size_t batch,
            const size_t batch_stride, const size_t elem_stride,
            cudaStream_t stream=0);

    } // of namespace statistics
} // of namespace cudalib


#endif // cudalib_statistics_h
// end of file

