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

            namespace cusolverDn {
                const char * const capsule_t = "cusolverDn.handle";
            }

            namespace cufft {
                const char * const capsule_t = "cufft.plan";
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

// dtypes
// follow numpy definition
// copy from numpy/core/include/numpy/ndarraytypes.h
enum PYCUDA_TYPES {    PYCUDA_BOOL=0,
                    PYCUDA_BYTE, PYCUDA_UBYTE,
                    PYCUDA_SHORT, PYCUDA_USHORT,
                    PYCUDA_INT, PYCUDA_UINT,
                    PYCUDA_LONG, PYCUDA_ULONG,
                    PYCUDA_LONGLONG, PYCUDA_ULONGLONG,
                    PYCUDA_FLOAT, PYCUDA_DOUBLE, PYCUDA_LONGDOUBLE,
                    PYCUDA_CFLOAT, PYCUDA_CDOUBLE, PYCUDA_CLONGDOUBLE,
                    PYCUDA_OBJECT=17,
                    PYCUDA_STRING, PYCUDA_UNICODE,
                    PYCUDA_VOID,
                    /*
                     * New 1.6 types appended, may be integrated
                     * into the above in 2.0.
                     */
                    PYCUDA_DATETIME, PYCUDA_TIMEDELTA, PYCUDA_HALF,

                    PYCUDA_NTYPES,
                    PYCUDA_NOTYPE,
                    PYCUDA_CHAR,
                    PYCUDA_USERDEF=256,  /* leave room for characters */

                    /* The number of types not including the new 1.6 types */
                    PYCUDA_NTYPES_ABI_COMPATIBLE=21
};
// the cudaDataType definition
//typedef enum cudaDataType_t
//{
//	CUDA_R_16F= 2,  /* real as a half */
//	CUDA_C_16F= 6,  /* complex as a pair of half numbers */
//	CUDA_R_32F= 0,  /* real as a float */
//	CUDA_C_32F= 4,  /* complex as a pair of float numbers */
//	CUDA_R_64F= 1,  /* real as a double */
//	CUDA_C_64F= 5,  /* complex as a pair of double numbers */
//	CUDA_R_8I = 3,  /* real as a signed char */
//	CUDA_C_8I = 7,  /* complex as a pair of signed char numbers */
//	CUDA_R_8U = 8,  /* real as a unsigned char */
//	CUDA_C_8U = 9,  /* complex as a pair of unsigned char numbers */
//	CUDA_R_32I= 10, /* real as a signed int */
//	CUDA_C_32I= 11, /* complex as a pair of signed int numbers */
//	CUDA_R_32U= 12, /* real as a unsigned int */
//	CUDA_C_32U= 13  /* complex as a pair of unsigned int numbers */
//} cudaDataType;
#endif

// end of file
