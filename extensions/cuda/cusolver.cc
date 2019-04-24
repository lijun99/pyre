// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#include <portinfo>
#include <Python.h>
//#include <pyre/journal.h>
#include <iostream>
#include <sstream>

// my declarations
#include "capsules.h"
#include "cublas.h"

// local support
#include "vector.h"
#include "matrix.h"
//#include "exceptions.h"
#include "dtypes.h"

// access to cudalib definitions
#include <pyre/cuda.h>

// PyErr Object
namespace pyre { namespace extensions { namespace cuda { namespace cublas {
    
    PyObject * PycublasErr = nullptr;

} } } }


// raise exception
const char *
pyre::extensions::cuda::cublas::
cublasGetErrMsg(cublasStatus_t err)
{
    /*
    CUBLAS_STATUS_SUCCESS         =0,
    CUBLAS_STATUS_NOT_INITIALIZED =1,
    CUBLAS_STATUS_ALLOC_FAILED    =3,
    CUBLAS_STATUS_INVALID_VALUE   =7,
    CUBLAS_STATUS_ARCH_MISMATCH   =8,
    CUBLAS_STATUS_MAPPING_ERROR   =11,
    CUBLAS_STATUS_EXECUTION_FAILED=13,
    CUBLAS_STATUS_INTERNAL_ERROR  =14,
    CUBLAS_STATUS_NOT_SUPPORTED   =15,
    CUBLAS_STATUS_LICENSE_ERROR   =16
     */
    const char * message;
    switch(err) {
    case CUBLAS_STATUS_NOT_INITIALIZED:
        message = "cublas not initialized";
        break;
    case CUBLAS_STATUS_ALLOC_FAILEDD:
        message = "CUBLAS_STATUS_ALLOC_FAILED";
        break;
    case CUBLAS_STATUS_INVALID_VALUE:
        message = "CUBLAS_STATUS_INVALID_VALUE";
        break;
    case CUBLAS_STATUS_ARCH_MISMATCHR:
        message = "CUBLAS_STATUS_ARCH_MISMATCH";
        break;
    case CUBLAS_STATUS_MAPPING_ERROR:
        message = "CUBLAS_STATUS_MAPPING_ERROR";
        break;
    case CUBLAS_STATUS_EXECUTION_FAILED:
        message = "CUBLAS_STATUS_EXECUTION_FAILED";
        break;
    case CUBLAS_STATUS_INTERNAL_ERROR:
        message = "CUBLAS_STATUS_INTERNAL_ERROR";
        break;
    case CUBLAS_STATUS_NOT_SUPPORTED:
        message = "CUBLAS_STATUS_NOT_SUPPORTED ";
        break;
    case CUBLAS_STATUS_LICENSE_ERROR:
        message = "CUBLAS_STATUS_LICENSE_ERROR";
        break;
    default:
        message = "unknown cublas error";
    }
    return message;
}


const char * const pyre::extensions::cuda::cublas::registerExceptions__name__ = "cublas_registerExceptions";
const char * const pyre::extensions::cuda::cublas::registerExceptions__doc__ = "register python cublas exception";

PyObject *
pyre::extensions::cuda::cublas::
registerExceptions(PyObject * module, PyObject * args)
{

    // unpack the arguments
    PyObject * exceptions;
    if (!PyArg_ParseTuple(args, "O!:cublas_registerExceptions", &PyModule_Type, &exceptions)) {
        return nullptr;
    }
    
    // create the cublas exception
    PycublasErr = PyErr_NewException("cublas_error", NULL, NULL);

    Py_INCREF(PycublasErr);
    // register the base class
    PyModule_AddObject(module, "cublasError", PycublasErr);

    // and return the module
    Py_INCREF(Py_None);
    return Py_None;
}

// allocate generator
const char * const pyre::extensions::cuda::cublas::alloc__name__ = "cublas_alloc";
const char * const pyre::extensions::cuda::cublas::alloc__doc__ = "allocate a cublas handle";
PyObject *
pyre::extensions::cuda::cublas::
alloc(PyObject *, PyObject *args)
{
    // create a cublas generator 
    cublasHandle_t handle = NULL;
    cublasSafeCall(cublasCreate(&handle);

    // return as a capsule
    return PyCapsule_New(handle, capsule_t, free);
}

// default destructor
void
pyre::extensions::cuda::cublas::
free(PyObject * capsule)
{
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) return;
    // get the generator
    cublasHandle_t handle =
        static_cast<cublasGenerator_t>(PyCapsule_GetPointer(capsule, capsule_t));
    
    // deallocate
    cublasSafeCall(cublasDestroy(handle));
    // and return
    return;
}

//end of file
