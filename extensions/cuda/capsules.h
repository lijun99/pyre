// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//


#if !defined(pyre_extensions_cuda_capsules_h)
#define pyre_extensions_cuda_capsules_h

// capsules
namespace pyre {
    namespace extensions {
        namespace cuda {
            // memory
            namespace vector {
                const char * const capsule_t = "cuda.vector";
            }
            namespace matrix {
                const char * const capsule_t = "cuda.matrix";
            }

            namespace cublas {
                const char * const capsule_t = "cublas.handle";
            }

            namespace curand {
                const char * const capsule_t = "curand.generator";
            }
            
            namespace stream {
               const char * const capsule_t = "cuda.stream";
            }

            namespace timer {
                const char * const capsule_t = "cuda.timer";
            }
            

        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
