// -*- C++ -*-
//
// michael a.g. aïvázis, orthologue
// lijun zhu, caltech
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <Python.h>

// journal
#include <pyre/journal.h>

// CUDA
#include <cuda.h>

// boilerplate
#include "metadata.h"
#include "exceptions.h"

// device management
#include "device.h"
// the module method declarations
#include "discover.h"

//
#include "vector.h"
#include "matrix.h"
#include "gsl.h"
#include "numpy.h"
#include "curand.h"
#include "cublas.h"
#include "cusolver.h"
#include "stream.h"
#include "timer.h"

// put everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {
            // the module method table
            PyMethodDef methods[] = {
                // module metadata
                // copyright
                { copyright__name__, copyright, METH_VARARGS, copyright__doc__ },
                // version
                { version__name__, version, METH_VARARGS, version__doc__ },
                // license
                { license__name__, license, METH_VARARGS, license__doc__ },
                
                // registration
                { registerExceptions__name__,
                  registerExceptions, METH_VARARGS, registerExceptions__doc__ },

                // device management
                { setDevice__name__, setDevice, METH_VARARGS, setDevice__doc__ },
                { resetDevice__name__, resetDevice, METH_VARARGS, resetDevice__doc__ },
                { synchronizeDevice__name__, synchronizeDevice, METH_VARARGS, synchronizeDevice__doc__ },

                // device discovery and other administrative tasks
                // discover
                { discover__name__, discover, METH_VARARGS, discover__doc__ },


                // vector
                {vector::alloc__name__, vector::alloc, METH_VARARGS, vector::alloc__doc__},
                {vector::zero__name__, vector::zero, METH_VARARGS, vector::zero__doc__},
                {vector::fill__name__, vector::fill, METH_VARARGS, vector::fill__doc__},
                {vector::copy__name__, vector::copy, METH_VARARGS, vector::copy__doc__},
                {vector::iadd__name__, vector::iadd, METH_VARARGS, vector::iadd__doc__},
                {vector::isub__name__, vector::isub, METH_VARARGS, vector::isub__doc__},
                {vector::imul__name__, vector::imul, METH_VARARGS, vector::imul__doc__},

                // matrix
                {matrix::alloc__name__, matrix::alloc, METH_VARARGS, matrix::alloc__doc__},
                {matrix::zero__name__, matrix::zero, METH_VARARGS, matrix::zero__doc__},
                {matrix::fill__name__, matrix::fill, METH_VARARGS, matrix::fill__doc__},
                {matrix::iadd__name__, matrix::iadd, METH_VARARGS, matrix::iadd__doc__},
                {matrix::isub__name__, matrix::isub, METH_VARARGS, matrix::isub__doc__},
                {matrix::imul__name__, matrix::imul, METH_VARARGS, matrix::imul__doc__},
                {matrix::copy__name__, matrix::copy, METH_VARARGS, matrix::copy__doc__},
                {matrix::copytile__name__, matrix::copytile, METH_VARARGS, matrix::copytile__doc__},
                {matrix::copycols__name__, matrix::copycols, METH_VARARGS, matrix::copycols__doc__},
                {matrix::duplicate_vector__name__, matrix::duplicate_vector, METH_VARARGS, matrix::duplicate_vector__doc__},
                {matrix::transpose__name__, matrix::transpose, METH_VARARGS, matrix::transpose__doc__},
                {matrix::inverse__name__, matrix::inverse, METH_VARARGS, matrix::inverse__doc__},
                {matrix::inverse_lu_cusolver__name__, matrix::inverse_lu_cusolver, METH_VARARGS, matrix::inverse_lu_cusolver__doc__},
                {matrix::inverse_cholesky__name__, matrix::inverse_cholesky, METH_VARARGS, matrix::inverse_cholesky__doc__},
                {matrix::cholesky__name__, matrix::cholesky, METH_VARARGS, matrix::cholesky__doc__},
                {matrix::determinant_triangular__name__, matrix::determinant_triangular, METH_VARARGS, matrix::determinant_triangular__doc__},
                {matrix::logdet_triangular__name__, matrix::logdet_triangular, METH_VARARGS, matrix::logdet_triangular__doc__},
                {matrix::determinant__name__, matrix::determinant, METH_VARARGS, matrix::determinant__doc__},

                // gsl hook
                {vector::togsl__name__, vector::togsl, METH_VARARGS, vector::togsl__doc__},
                {vector::fromgsl__name__, vector::fromgsl, METH_VARARGS, vector::fromgsl__doc__},
                {matrix::togsl__name__, matrix::togsl, METH_VARARGS, matrix::togsl__doc__},
                {matrix::fromgsl__name__, matrix::fromgsl, METH_VARARGS, matrix::fromgsl__doc__},

                // numpy hook
                {vector::tonumpy__name__, vector::tonumpy, METH_VARARGS, vector::tonumpy__doc__},
                {vector::fromnumpy__name__, vector::fromnumpy, METH_VARARGS, vector::fromnumpy__doc__},
                {matrix::tonumpy__name__, matrix::tonumpy, METH_VARARGS, matrix::tonumpy__doc__},
                {matrix::fromnumpy__name__, matrix::fromnumpy, METH_VARARGS, matrix::fromnumpy__doc__},

                // curand
                {curand::registerExceptions__name__, curand::registerExceptions, METH_VARARGS, curand::registerExceptions__doc__},
                {curand::alloc__name__, curand::alloc, METH_VARARGS, curand::alloc__doc__},
                {curand::setseed__name__, curand::setseed, METH_VARARGS, curand::setseed__doc__},
                {curand::gaussian__name__, curand::gaussian, METH_VARARGS, curand::gaussian__doc__},
                {curand::uniform__name__, curand::uniform, METH_VARARGS, curand::uniform__doc__},

                // cublas
                {cublas::registerExceptions__name__, cublas::registerExceptions, METH_VARARGS, cublas::registerExceptions__doc__},
                {cublas::alloc__name__, cublas::alloc, METH_VARARGS, cublas::alloc__doc__},
                {cublas::axpy__name__, cublas::axpy, METH_VARARGS, cublas::axpy__doc__},
                {cublas::nrm2__name__, cublas::nrm2, METH_VARARGS, cublas::nrm2__doc__},
                {cublas::trmv__name__, cublas::trmv, METH_VARARGS, cublas::trmv__doc__},
                {cublas::trmm__name__, cublas::trmm, METH_VARARGS, cublas::trmm__doc__},
                {cublas::gemm__name__, cublas::gemm, METH_VARARGS, cublas::gemm__doc__},
                {cublas::symv__name__, cublas::symv, METH_VARARGS, cublas::symv__doc__},
                {cublas::syr__name__, cublas::syr, METH_VARARGS, cublas::syr__doc__},

                // cusolver
                {cusolverDn::cusolverDnCreate__name__, cusolverDn::cusolverDnCreate, METH_VARARGS, cusolverDn::cusolverDnCreate__doc__},
                {cusolverDn::cusolverDnSetStream__name__, cusolverDn::cusolverDnSetStream, METH_VARARGS, cusolverDn::cusolverDnSetStream__doc__},

                // stream
                {stream::alloc__name__, stream::alloc, METH_VARARGS, stream::alloc__name__},

                // timer
                {timer::alloc__name__, timer::alloc, METH_VARARGS, timer::alloc__name__},
                {timer::start__name__, timer::start, METH_VARARGS, timer::start__name__},
                {timer::stop__name__, timer::stop, METH_VARARGS, timer::stop__name__},
                {timer::time__name__, timer::time, METH_VARARGS, timer::time__name__},


                // sentinel
                {0, 0, 0, 0}
            };


            // the module documentation string
            const char * const doc = "provides access to CUDA enabled devices";

            // the module definition structure
            PyModuleDef module = {
                // header
                PyModuleDef_HEAD_INIT,
                // the name of the module
                "cuda",
                // the module documentation string
                doc,
                // size of the per-interpreter state of the module; -1 if this state is global
                -1,
                // the methods defined in this module
                methods
            };

        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre

// initialization function for the module
// *must* be called PyInit_cuda
PyMODINIT_FUNC
PyInit_cuda()
{
    // create the module
    PyObject * module = PyModule_Create(&pyre::extensions::cuda::module);
    // check whether module creation succeeded and raise an exception if not
    if (!module) {
        return module;
    }

#ifdef USE_CUDA_CRIVER_API
    // initialize cuda
    CUresult status = cuInit(0);
    // check
    if (status != CUDA_SUCCESS) {
        // something went wrong
        PyErr_SetString(PyExc_ImportError, "CUDA initialization failed");
        // raise an exception
        return 0;
    }
#endif

    //pyre::extensions::cuda::curand::PyCurandErr = PyErr_NewException("CURANDError", NULL, NULL);
    //Py_INCREF(pyre::extensions::cuda::curand::PyCurandErr);
    //PyModule_AddObject(module, "error", pyre::extensions::cuda::curand::PyCurandErr);

    // and return the newly created module
    return module;
}


// end of file
