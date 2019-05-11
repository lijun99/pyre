// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#if !defined(pyre_extensions_cuda_matrix_h)
#define pyre_extensions_cuda_matrix_h

// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {
            // matrix
            namespace matrix {
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

                  // copy a tile submatrix or insertion
                  extern const char * const copytile__name__;
                  extern const char * const copytile__doc__;
                  PyObject * copytile(PyObject *, PyObject *);

                  // copy selected cols with indices
                  extern const char * const copycols__name__;
                  extern const char * const copycols__doc__;
                  PyObject * copycols(PyObject *, PyObject *);

                  // duplicate a vector
                  extern const char * const duplicate_vector__name__;
                  extern const char * const duplicate_vector__doc__;
                  PyObject * duplicate_vector(PyObject *, PyObject *);

                  // transpose
                  extern const char * const transpose__name__;
                  extern const char * const transpose__doc__;
                  PyObject * transpose(PyObject *, PyObject *);

                  // inverse
                  extern const char * const inverse__name__;
                  extern const char * const inverse__doc__;
                  PyObject * inverse(PyObject *, PyObject *);

                  // inverse_lu
                  extern const char * const inverse_lu_cusolver__name__;
                  extern const char * const inverse_lu_cusolver__doc__;
                  PyObject * inverse_lu_cusolver(PyObject *, PyObject *);

                  // inverse_symm
                  extern const char * const inverse_cholesky__name__;
                  extern const char * const inverse_cholesky__doc__;
                  PyObject * inverse_cholesky(PyObject *, PyObject *);

                  // cholesky factorization
                  extern const char * const cholesky__name__;
                  extern const char * const cholesky__doc__;
                  PyObject * cholesky(PyObject *, PyObject *);

                  // determinant_triangular
                  extern const char * const determinant_triangular__name__;
                  extern const char * const determinant_triangular__doc__;
                  PyObject * determinant_triangular(PyObject *, PyObject *);

                  // logdet_triangular
                  extern const char * const logdet_triangular__name__;
                  extern const char * const logdet_triangular__doc__;
                  PyObject * logdet_triangular(PyObject *, PyObject *);

                  // determinant
                  extern const char * const determinant__name__;
                  extern const char * const determinant__doc__;
                  PyObject * determinant(PyObject *, PyObject *);


            } // of namespace matrix
        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
