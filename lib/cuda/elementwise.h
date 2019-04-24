// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#ifndef cudalib_elementwise_h
#define cudalib_elementwise_h

// common elementwise cuda routines

namespace cudalib {
    namespace elementwise {
        // set values (use cudaMemset instead for 0)
        template<typename T>
            void fill(T *array, const size_t n, const T value, cudaStream_t stream=0);
        
        // copy data from T2 type to T1 type  
        template<typename T1, typename T2> 
            void conversion(T1 *out, const T2 *in, const size_t n, cudaStream_t stream=0);
        
        // add a1 += a2 
        template<typename T>
            void iadd(T *a1, const T* a2, const size_t n, cudaStream_t stream=0);

        // sub a1 -= a2 
        template<typename T>
            void isub(T *a1, const T* a2, const size_t n, cudaStream_t stream=0);

        // mul a1 *= a2 (a2 scalar) 
        template<typename T>
            void imul(T *a1, const T a2, const size_t n, cudaStream_t stream=0);
            
    } // of namespace elementwise
} // of namespace cudalib

#endif // cudalib_elementwise_h
// end of file
