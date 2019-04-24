// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#if !defined(cuda_extensions_vector_h)
#define cuda_extensions_vector_h

// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {
            // vector
            namespace vector {
                  // allocate
                  extern const char * const alloc__name__;
                  extern const char * const alloc__doc__;
                  PyObject * alloc(PyObject *, PyObject *);

                  // free
                  void free(PyObject *);

                  // set_zero
                  extern const char * const zero__name__;
                  extern const char * const zero__doc__;
                  PyObject * zero(PyObject *, PyObject *);

                  // set_all
                  extern const char * const fill__name__;
                  extern const char * const fill__doc__;
                  PyObject * fill(PyObject *, PyObject *);

                  // iadd
                  extern const char * const iadd__name__;
                  extern const char * const iadd__doc__;
                  PyObject * iadd(PyObject *, PyObject *);

                  // isub
                  extern const char * const isub__name__;
                  extern const char * const isub__doc__;
                  PyObject * isub(PyObject *, PyObject *);

                  // imul
                  extern const char * const imul__name__;
                  extern const char * const imul__doc__;
                  PyObject * imul(PyObject *, PyObject *);
                  
                  // copy
                  extern const char * const copy__name__;
                  extern const char * const copy__doc__;
                  PyObject * copy(PyObject *, PyObject *);
                  
 
            } // of namespace vector
        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
