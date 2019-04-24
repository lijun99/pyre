// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#if !defined(cuda_extensions_curand_h)
#define cuda_extensions_curand_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {
            // curand
            namespace curand {
                // exception

                extern PyObject * PyCurandErr;

                extern const char * const registerExceptions__name__;
                extern const char * const registerExceptions__doc__;
                PyObject * registerExceptions(PyObject *, PyObject*);
                
                // allocate a curand generator
                extern const char * const alloc__name__;
                extern const char * const alloc__doc__;
                PyObject * alloc(PyObject *, PyObject *);

                // default deallocator        
                void free(PyObject *); 

                // set seed
                extern const char * const setseed__name__;
                extern const char * const setseed__doc__;
                PyObject * setseed(PyObject *, PyObject *);

                // gaussian pdf
                extern const char * const gaussian__name__;
                extern const char * const gaussian__doc__;
                PyObject * gaussian(PyObject *, PyObject *);
                
                // uniform pdf
                extern const char * const uniform__name__;
                extern const char * const uniform__doc__;
                PyObject * uniform(PyObject *, PyObject *);
                
            } // of namespace curand
        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre


#endif
// end of file
