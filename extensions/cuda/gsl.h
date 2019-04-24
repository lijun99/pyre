// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#if !defined(pyre_cuda_extensions_gsl_h)
#define pyre_cuda_extensions_gsl_h

// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {
            // vector
            namespace vector {
                  
                  // togsl
                  extern const char * const togsl__name__;
                  extern const char * const togsl__doc__;
                  PyObject * togsl(PyObject *, PyObject *);
                  
                  // fromgsl
                  extern const char * const fromgsl__name__;
                  extern const char * const fromgsl__doc__;
                  PyObject * fromgsl(PyObject *, PyObject *);

            } // of namespace vector
            
            namespace matrix {
                  // togsl
                  extern const char * const togsl__name__;
                  extern const char * const togsl__doc__;
                  PyObject * togsl(PyObject *, PyObject *);
                  
                  // fromgsl
                  extern const char * const fromgsl__name__;
                  extern const char * const fromgsl__doc__;
                  PyObject * fromgsl(PyObject *, PyObject *);

            } // of namespace matrix
            
        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
