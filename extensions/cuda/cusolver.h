// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#if !defined(cuda_extensions_cusolver_h)
#define cuda_extensions_cusolver_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {
            // cusolver
            namespace cusolverDn {

                // allocate a cusolverDN (dense matrix) handle
                extern const char * const cusolverDnCreate__name__;
                extern const char * const cusolverDnCreate__doc__;
                PyObject * cusolverDnCreate(PyObject *, PyObject *);

                // default deallocator
                void cusolverDnDestroy(PyObject *);

                // set stream
                extern const char * const cusolverDnSetStream__name__;
                extern const char * const cusolverDnSetStream__doc__;
                PyObject * cusolverDnSetStream(PyObject *, PyObject *);

            } // of namespace cusolver
        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre


#endif
// end of file
