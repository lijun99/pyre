// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#if !defined(cuda_extensions_cublas_h)
#define cuda_extensions_cublas_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {
            // cublas
            namespace cublas {
                // exception

                extern PyObject * PycublasErr;

                extern const char * const registerExceptions__name__;
                extern const char * const registerExceptions__doc__;
                PyObject * registerExceptions(PyObject *, PyObject*);

                // allocate a cublas handle
                extern const char * const alloc__name__;
                extern const char * const alloc__doc__;
                PyObject * alloc(PyObject *, PyObject *);

                // default deallocator
                void free(PyObject *);

                // axpy y = y + a x
                extern const char * const axpy__name__;
                extern const char * const axpy__doc__;
                PyObject * axpy(PyObject *, PyObject *);

                //  Euclidean(L2) norm ||x||
                extern const char * const nrm2__name__;
                extern const char * const nrm2__doc__;
                PyObject * nrm2(PyObject *, PyObject *);

                // triangular matrix-vector product
                extern const char * const trmv__name__;
                extern const char * const trmv__doc__;
                PyObject * trmv(PyObject *, PyObject *);

                // triangular matrix-matrix product
                extern const char * const trmm__name__;
                extern const char * const trmm__doc__;
                PyObject * trmm(PyObject *, PyObject *);

                //  matrix-matrix product
                extern const char * const gemm__name__;
                extern const char * const gemm__doc__;
                PyObject * gemm(PyObject *, PyObject *);

                //  matrix-vector product
                extern const char * const gemv__name__;
                extern const char * const gemv__doc__;
                PyObject * gemv(PyObject *, PyObject *);

                //  symmetric matrix matrix product
                extern const char * const symm__name__;
                extern const char * const symm__doc__;
                PyObject * symm(PyObject *, PyObject *);

                //  symmetric matrix vector product
                extern const char * const symv__name__;
                extern const char * const symv__doc__;
                PyObject * symv(PyObject *, PyObject *);

                //  symmetric rank-1 update
                extern const char * const syr__name__;
                extern const char * const syr__doc__;
                PyObject * syr(PyObject *, PyObject *);

            } // of namespace cublas
        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre


#endif
// end of file
