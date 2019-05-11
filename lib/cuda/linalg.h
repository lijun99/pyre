// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

/// @file linalg.h
///
/// @brief cuda Linear algebra routines, including matrix inverse, LU/Cholesky factorizations

#ifndef cudalib_linalg_h
#define cudalib_linalg_h

#include <cublas_v2.h>
#include <cusolverDn.h>


namespace cudalib {
    namespace linalg {

        // inverse of a matrix of (nxn)
        // inplace, make a copy if to conserve original matrix
        // cublas version doesn't work for large matrices (a bug?)
        template<typename T>
            int inverse_lu_cublas(cublasHandle_t handle, T * const matrix, const size_t n, cudaStream_t stream=0);

        // lu with cusolver
        template<typename T>
            int lu(cusolverDnHandle_t handle, T * const matrix,
                    const size_t lda, const size_t m, const size_t n, cudaStream_t stream=0);

        // inverse by cusolver lu
        template<typename T>
            int inverse_lu_cusolver(cusolverDnHandle_t handle, T * const matrix, const size_t n, cudaStream_t stream=0);

        // symmetric positive definite matrix inverse with Cholesky factorization; fastest
        template<typename T>
            int inverse_cholesky(cusolverDnHandle_t handle, T * const matrix, cublasFillMode_t uplo, const size_t n,
                            cudaStream_t stream=0);

        // Chelosky factorization of a matrix of (nxn)
        // inplace, make a copy if to conserve original matrix
        template<typename T>
            int cholesky(cusolverDnHandle_t handle, T * const matrix, cublasFillMode_t uplo,
                        const size_t n, cudaStream_t stream=0);

        // Determinant of a triangular matrix
        template<typename T>
            T determinant_triangular(const T * const mat, const size_t n, cudaStream_t stream=0);
        // log version is preferred in case of overflow
        template<typename T>
            T logdet_triangular(const T * const mat, const size_t n, cudaStream_t stream=0);

        // Determinant of a matrix through Cholesky
        // inplace, make a copy if to conserve original matrix
        template<typename T>
            T determinant_cusolver(cusolverDnHandle_t handle, T * const mat, const size_t n, cudaStream_t stream=0);

    } // of namespace linalg
} // of namespace cudalib

#endif // cudalib_linalg_h
// end of file
