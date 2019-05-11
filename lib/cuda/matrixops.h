// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

/// @file matrixops_h
/// @brief provide several matrix operations

#ifndef cudalib_matrixops_h
#define cudalib_matrixops_h


namespace cudalib {
    namespace matrix {

        //  create a nxn unit/identity matrix
        template<typename T>
        void identity(T * const m, const size_t n, cudaStream_t stream=0);

        // view (a submatrix) - copy a tile of matrix
        template<typename Tout, typename Tin>
        void copy_tile(Tout * const odata,  const size_t ldo, // the memory address and leading dimension of output matrix
                const size_t omstart, const size_t onstart, // starting position (row, col) of output
                const Tin * const idata, const size_t ldi, // the memory address and leading dimension of input matrix
                const size_t imstart, const size_t instart, // starting position of idata
                const size_t m, const size_t n, // tile size to be copied
                cudaStream_t stream=0);

        // copy matrix with selected columns
        template<typename T>
        void copy_indices(T * const odata,  const size_t ldo,
                const T * const idata, const size_t ldi,
                const size_t m, const size_t n, // tile to be copied
                const size_t * const indices, // list of col indices
                cudaStream_t stream=0);

        // transpose  idata (iM, iN) -> odata (iN, iM)
        template<typename T>
        void transpose(T * const odata, const T* const idata, // output/input memory address
                const size_t iM, const size_t iN, // size of input
                cudaStream_t stream=0);

        // duplicate a vector into rows in a matrix
        template<typename T>
        void duplicate_vector(T* const odata,  const size_t ldo, // output matrix
                const T* const idata, const size_t incx,
                const size_t m, const size_t n,
                cudaStream_t stream=0);

    } // of namespace matrix
} // of namespace cudalib

#endif // cudalib_matrixops_h
// end of file
