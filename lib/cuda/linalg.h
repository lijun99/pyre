// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#ifndef cudalib_linalg_h
#define cudalib_linalg_h

#include <cublas_v2.h>
#include <cusolverDn.h>


// common linalg cuda routines

namespace cudalib {
    namespace linalg {
 
        // inverse of a matrix of (nxn)
        // inplace, make a copy if to conserve original matrix
        // cublas version doesn't  work for large matrices
        template<typename T>
            int inverse_cublas(cublasHandle_t handle, T * const matrix, const size_t n, cudaStream_t stream=0);

        //  LU ; not finished yet
        template<typename T>
            int lu(T * const matrix, const size_t lda, const size_t m, const size_t n, cudaStream_t stream=0);

        // symmetric matrix inverse with Cholesky factorization
        int inverse_symm_S(float * matrix, cublasFillMode_t uplo, const size_t n, cudaStream_t stream=0);
        int inverse_symm_D(double * matrix, cublasFillMode_t uplo, const size_t n, cudaStream_t stream=0);

        // Chelosky factorization of a matrix of (nxn)
        // inplace, make a copy if to conserve original matrix
        template<typename T>
            int cholesky(T * const matrix, cublasFillMode_t uplo, const size_t n, cudaStream_t stream=0);

        // Determinant of a triangular matrix
        template<typename T>
            T determinant_triangular(T * const mat, const size_t n, cudaStream_t stream=0);
        template<typename T>
            T logdet_triangular(T * const mat, const size_t n, cudaStream_t stream=0);

        // Determinant of a matrix through cholesky
        // inplace, make a copy if to conserve original matrix
        template<typename T>
            T determinant(T * const mat, const size_t n, cudaStream_t stream=0);
        
            
    } // of namespace linalg
} // of namespace cudalib

#endif // cudalib_linalg_h
// end of file
