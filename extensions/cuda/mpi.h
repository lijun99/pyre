// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#if !defined(pyre_cuda_extensions_mpi_h)
#define pyre_cuda_extensions_mpi_h

// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {
            // vector
            namespace vector {

                  // bcast
                  extern const char * const bcast__name__;
                  extern const char * const bcast__doc__;
                  PyObject * bcast(PyObject *, PyObject *);
            }

            namespace matrix {

                  // bcast
                  extern const char * const bcast__name__;
                  extern const char * const bcast__doc__;
                  PyObject * bcast(PyObject *, PyObject *);
            }

        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
