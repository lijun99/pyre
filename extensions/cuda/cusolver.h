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

                
            } // of namespace cublas
        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre


#endif
// end of file
