// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#if !defined(cuda_extensions_stream_h)
#define cuda_extensions_stream_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {
            // cublas
            namespace stream {
                // exception
                
                // allocate a cublas handle
                extern const char * const alloc__name__;
                extern const char * const alloc__doc__;
                PyObject * alloc(PyObject *, PyObject *);

                // default deallocator        
                void free(PyObject *); 

                
            } // of namespace stream
        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre


#endif
// end of file
