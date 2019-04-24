// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#if !defined(cuda_extensions_cutimer_h)
#define cuda_extensions_cutimer_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {
            // cutimer
            namespace timer {
                // exception
                
                // allocate a cutimer generator
                extern const char * const alloc__name__;
                extern const char * const alloc__doc__;
                PyObject * alloc(PyObject *, PyObject *);

                // default deallocator        
                void free(PyObject *); 

                // start
                extern const char * const start__name__;
                extern const char * const start__doc__;
                PyObject * start(PyObject *, PyObject *);

                // stop
                extern const char * const stop__name__;
                extern const char * const stop__doc__;
                PyObject * stop(PyObject *, PyObject *);
                
                // time
                extern const char * const time__name__;
                extern const char * const time__doc__;
                PyObject * time(PyObject *, PyObject *);
                
            } // of namespace cutimer
        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre


#endif
// end of file
