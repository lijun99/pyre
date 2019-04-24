// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#ifndef cudalib_matrixops_h
#define cudalib_matrixops_h

// common matrix cuda routines

namespace cudalib {
    namespace matrix {
        // view (a submatrix)
        template<typename Tout, typename Tin>
        void copy_tile(Tout * const odata,  const size_t ldo, 
                const size_t omstart, const size_t onstart, // starting position of odata
                const Tin * const idata, const size_t ldi,
                const size_t imstart, const size_t instart, // starting position of idata
                const size_t m, const size_t n, // tile to be copied
                cudaStream_t stream=0);

        // copy matrix with selected cols
        template<typename T>
        void copy_indices(T * const odata,  const size_t ldo, 
                const T * const idata, const size_t ldi,
                const size_t m, const size_t n, // tile to be copied
                const size_t * const indices, // list of col indices
                cudaStream_t stream=0);

        // transpose
        template<typename T>
        void transpose(T * const odata, const T* const idata, const size_t iM, const size_t iN, cudaStream_t stream=0);

        // duplicate vector
        template<typename T>
        void duplicate_vector(T* const odata,  const size_t ldo, const T* const idata, const size_t incx,
                const size_t m, const size_t n, cudaStream_t stream=0);


    } // of namespace matrix
} // of namespace cudalib

#endif // cudalib_matrixops_h
// end of file
