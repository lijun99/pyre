// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#if !defined(pyre_extensions_cuda_stats_h)
#define pyre_extensions_cuda_stats_h

// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {
            // stats
            namespace stats {

                // vector_amin
                extern const char * const vector_amin__name__;
                extern const char * const vector_amin__doc__;
                PyObject * vector_amin(PyObject *, PyObject *);

                // vector_amax
                extern const char * const vector_amax__name__;
                extern const char * const vector_amax__doc__;
                PyObject * vector_amax(PyObject *, PyObject *);

                // vector_sum
                extern const char * const vector_sum__name__;
                extern const char * const vector_sum__doc__;
                PyObject * vector_sum(PyObject *, PyObject *);

                // vector_mean
                extern const char * const vector_mean__name__;
                extern const char * const vector_mean__doc__;
                PyObject * vector_mean(PyObject *, PyObject *);

                // vector_std
                extern const char * const vector_std__name__;
                extern const char * const vector_std__doc__;
                PyObject * vector_std(PyObject *, PyObject *);

                // matrix_amin
                extern const char * const matrix_amin__name__;
                extern const char * const matrix_amin__doc__;
                PyObject * matrix_amin(PyObject *, PyObject *);

                // matrix_amax
                extern const char * const matrix_amax__name__;
                extern const char * const matrix_amax__doc__;
                PyObject * matrix_amax(PyObject *, PyObject *);

                // matrix_sum
                extern const char * const matrix_sum__name__;
                extern const char * const matrix_sum__doc__;
                PyObject * matrix_sum(PyObject *, PyObject *);

                // matrix_mean
                extern const char * const matrix_mean_flattened__name__;
                extern const char * const matrix_mean_flattened__doc__;
                PyObject * matrix_mean_flattened(PyObject *, PyObject *);

                extern const char * const matrix_mean__name__;
                extern const char * const matrix_mean__doc__;
                PyObject * matrix_mean(PyObject *, PyObject *);

                // matrix_mean_std
                extern const char * const matrix_mean_std__name__;
                extern const char * const matrix_mean_std__doc__;
                PyObject * matrix_mean_std(PyObject *, PyObject *);

                // vector_covariance
                extern const char * const vector_covariance__name__;
                extern const char * const vector_covariance__doc__;
                PyObject * vector_covariance(PyObject *, PyObject *);

                // vector_correlation
                extern const char * const vector_correlation__name__;
                extern const char * const vector_correlation__doc__;
                PyObject * vector_correlation(PyObject *, PyObject *);

                // matrix_covariance
                extern const char * const matrix_covariance__name__;
                extern const char * const matrix_covariance__doc__;
                PyObject * matrix_covariance(PyObject *, PyObject *);

                // matrix_correlation
                extern const char * const matrix_correlation__name__;
                extern const char * const matrix_correlation__doc__;
                PyObject * matrix_correlation(PyObject *, PyObject *);

                // L1norm for a vector
                extern const char * const L1norm__name__;
                extern const char * const L1norm__doc__;
                PyObject * L1norm(PyObject *, PyObject *);

                // L2norm for a vector
                extern const char * const L2norm__name__;
                extern const char * const L2norm__doc__;
                PyObject * L2norm(PyObject *, PyObject *);

                // Linfnorm for a vector
                extern const char * const Linfnorm__name__;
                extern const char * const Linfnorm__doc__;
                PyObject * Linfnorm(PyObject *, PyObject *);



            } // of namespace stats
        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
