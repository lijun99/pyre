// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#if !defined(pyre_cuda_extensions_numpy_h)
#define pyre_cuda_extensions_numpy_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {
            // vector
            namespace vector {
                // vector_tonumpy
                extern const char * const tonumpy__name__;
                extern const char * const tonumpy__doc__;
                PyObject * tonumpy(PyObject *, PyObject *);
                
                // vector_fromnumpy
                extern const char * const fromnumpy__name__;
                extern const char * const fromnumpy__doc__;
                PyObject * fromnumpy(PyObject *, PyObject *);
                
            } // of namespace vector

            namespace matrix {
                // matrixnumpy
                extern const char * const tonumpy__name__;
                extern const char * const tonumpy__doc__;
                PyObject * tonumpy(PyObject *, PyObject *);

                extern const char * const fromnumpy__name__;
                extern const char * const fromnumpy__doc__;
                PyObject * fromnumpy(PyObject *, PyObject *);
            } // of namespace matrix
            
       } // of namespace cuda
    }// of namespace extensions
} // of namespace pyre

#endif //pyre_cuda_extensions_numpy_h

// end of file
