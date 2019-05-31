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
#include "curand.h"

// local support
#include "vector.h"
#include "matrix.h"
//#include "exceptions.h"
#include "dtypes.h"

// access to cudalib definitions
#include <pyre/cuda.h>

// PyErr Object
namespace pyre { namespace extensions { namespace cuda { namespace curand {
    const char * curandGetErrMsg(curandStatus_t err);

    PyObject * PyCurandErr = nullptr;
} } } }


// raise exception
const char *
pyre::extensions::cuda::curand::
curandGetErrMsg(curandStatus_t err)
{
    const char * message;
    switch(err) {
    case CURAND_STATUS_VERSION_MISMATCH:
        message = "Header file and linked library version do not match";
        break;
    case CURAND_STATUS_NOT_INITIALIZED:
        message = "Generator not initialized";
        break;
    case CURAND_STATUS_ALLOCATION_FAILED:
        message = "Memory allocation failed";
        break;
    case CURAND_STATUS_TYPE_ERROR:
        message = "Generator is wrong type";
        break;
    case CURAND_STATUS_OUT_OF_RANGE:
        message = "Argument out of range";
        break;
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        message = "Length requested is not a multple of dimension";
        break;
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        message = "GPU does not have double precision required by MRG32k3a";
        break;
    case CURAND_STATUS_LAUNCH_FAILURE:
        message = "Kernel launch failure";
        break;
    case CURAND_STATUS_PREEXISTING_FAILURE:
        message = "Preexisting failure on library entry";
        break;
    case CURAND_STATUS_INITIALIZATION_FAILED:
        message = "Initialization of CUDA failed";
        break;
    case CURAND_STATUS_ARCH_MISMATCH:
        message = "Architecture mismatch, GPU does not support requested feature";
        break;
    case CURAND_STATUS_INTERNAL_ERROR:
        message = "Internal library error";
        break;
    default:
        message = "other curand error";
    }
    return message;
}


const char * const pyre::extensions::cuda::curand::registerExceptions__name__ = "curand_registerExceptions";
const char * const pyre::extensions::cuda::curand::registerExceptions__doc__ = "register python curand exception";

PyObject *
pyre::extensions::cuda::curand::
registerExceptions(PyObject * module, PyObject * args)
{

    // unpack the arguments
    PyObject * exceptions;
    if (!PyArg_ParseTuple(args, "O!:curand_registerExceptions", &PyModule_Type, &exceptions)) {
        return nullptr;
    }

    // create the curand exception
    PyCurandErr = PyErr_NewException("curand_error", NULL, NULL);

    Py_INCREF(PyCurandErr);
    // register the base class
    PyModule_AddObject(module, "CURANDError", PyCurandErr);

    // and return the module
    Py_INCREF(Py_None);
    return Py_None;
}

// allocate generator
const char * const pyre::extensions::cuda::curand::alloc__name__ = "curand_alloc";
const char * const pyre::extensions::cuda::curand::alloc__doc__ = "allocate a curand generator";
PyObject *
pyre::extensions::cuda::curand::
alloc(PyObject *, PyObject *args)
{
    // allocate storage for the arguments
    int genType;
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "i:curand_alloc", &genType)) {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for curand.alloc");
        return nullptr;
    }

    // create a curand generator
    curandGenerator_t gen = NULL;
    curandSafeCall(curandCreateGenerator(&gen, (curandRngType_t)genType));
    /*if(status!=CURAND_STATUS_SUCCESS) {
        PyErr_SetString(PyCurandErr, curandGetErrMsg(status));
        return 0;
    }*/

    // return as a capsule
    return PyCapsule_New(gen, capsule_t, free);
}

// default destructor
void
pyre::extensions::cuda::curand::
free(PyObject * capsule)
{
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) return;
    // get the generator
    curandGenerator_t gen =
        static_cast<curandGenerator_t>(PyCapsule_GetPointer(capsule, capsule_t));

    // deallocate
    curandSafeCall(curandDestroyGenerator(gen));
    // and return
    return;
}


// set seed
const char * const pyre::extensions::cuda::curand::setseed__name__ = "curand_setseed";
const char * const pyre::extensions::cuda::curand::setseed__doc__ = "set seed for curand generator";
PyObject *
pyre::extensions::cuda::curand::
setseed(PyObject *, PyObject *args)
{
    // allocate storage for the arguments
    unsigned long long seed;
    PyObject * genCapsule; // curand generator capsule
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "O!K:curand_alloc", &PyCapsule_Type, &genCapsule, &seed)) {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for curand_setseed");
        return nullptr;
    }
    // check curand generator capsule
    if (!PyCapsule_IsValid(genCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid curand generator capsule");
        return 0;
    }
    // get the generator
    curandGenerator_t gen =
        static_cast<curandGenerator_t>(PyCapsule_GetPointer(genCapsule, capsule_t));
    // call
    curandSafeCall(curandSetPseudoRandomGeneratorSeed(gen, seed));

    // return None
    Py_RETURN_NONE;
}

// normal/gaussian distribution generation
const char * const pyre::extensions::cuda::curand::gaussian__name__ = "curand_gaussian";
const char * const pyre::extensions::cuda::curand::gaussian__doc__ = "curand generate gaussian distribution samples for vector/matrix";
PyObject *
pyre::extensions::cuda::curand::
gaussian(PyObject *, PyObject *args)
{
    // allocate storage for the arguments
    PyObject * genCapsule; // curand generator capsule
    PyObject * dataCapsule; // cuda vector or matrix
    double mean, sdev; // gaussian (mean, sdev)
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "O!O!dd:curand_gaussian", &PyCapsule_Type, &genCapsule, &PyCapsule_Type, &dataCapsule, &mean, &sdev)) {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for curand_gaussian");
        return nullptr;
    }
    // check curand generator capsule
    if (!PyCapsule_IsValid(genCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid curand generator capsule");
        return 0;
    }
    // get the generator
    curandGenerator_t gen =
        static_cast<curandGenerator_t>(PyCapsule_GetPointer(genCapsule, capsule_t));

    // check the data capsule type
    if (PyCapsule_IsValid(dataCapsule, pyre::extensions::cuda::vector::capsule_t)) {
        // get the vector
        cuda_vector * v = static_cast<cuda_vector *>(PyCapsule_GetPointer(dataCapsule, pyre::extensions::cuda::vector::capsule_t));

        switch(v->dtype) {
            case PYCUDA_DOUBLE:
                // cuda pseudorandom generator for Normal/logNormal requires even numbers
                if (v->size %2) {
                    double * v_temp;
                    size_t size_temp = v->size+1;
                    cudaSafeCall(cudaMalloc((void **)&v_temp, size_temp*sizeof(double)));
                    curandSafeCall(curandGenerateNormalDouble(gen, v_temp, size_temp, mean, sdev));
                    cudaSafeCall(cudaMemcpy(v->data, v_temp, v->nbytes, cudaMemcpyDefault));
                    cudaSafeCall(cudaFree(v_temp));
                }
                else {
                    curandSafeCall(curandGenerateNormalDouble(gen, (double *)v->data, v->size, mean, sdev));
                }
                break;
            case PYCUDA_FLOAT:
                if (v->size %2) {
                    float * v_temp;
                    size_t size_temp = v->size+1;
                    cudaSafeCall(cudaMalloc((void **)&v_temp, size_temp*sizeof(float)));
                    curandSafeCall(curandGenerateNormal(gen, v_temp, size_temp, (float)mean, (float)sdev));
                    cudaSafeCall(cudaMemcpy(v->data, v_temp, v->nbytes, cudaMemcpyDefault));
                    cudaSafeCall(cudaFree(v_temp));
                }
                else {
                    curandSafeCall(curandGenerateNormal(gen, (float *)v->data, v->size, (float)mean, (float)sdev));
                }
                break;
            default:
                PyErr_SetString(PyExc_TypeError, "invalid data type for curand generation");
                return 0;
        }
    }
    else if (PyCapsule_IsValid(dataCapsule, pyre::extensions::cuda::matrix::capsule_t)) {
        // get the matrix
        cuda_matrix * m = static_cast<cuda_matrix *>(PyCapsule_GetPointer(dataCapsule, pyre::extensions::cuda::matrix::capsule_t));
        // check the data type of matrix
        switch(m->dtype) {
            case PYCUDA_DOUBLE:
                // cuda pseudorandom generator for Normal/logNormal requires even numbers
                if (m->size %2) {
                    double * m_temp;
                    size_t size_temp = m->size+1;
                    cudaSafeCall(cudaMalloc((void **)&m_temp, size_temp*sizeof(double)));
                    curandSafeCall(curandGenerateNormalDouble(gen, m_temp, size_temp, mean, sdev));
                    cudaSafeCall(cudaMemcpy(m->data, m_temp, m->nbytes, cudaMemcpyDefault));
                    cudaSafeCall(cudaFree(m_temp));
                }
                else {
                    curandSafeCall(curandGenerateNormalDouble(gen, (double *)m->data, m->size, mean, sdev));
                }
                break;
            case PYCUDA_FLOAT:
                if (m->size %2) {
                    float * m_temp;
                    size_t size_temp = m->size+1;
                    cudaSafeCall(cudaMalloc((void **)&m_temp, size_temp*sizeof(float)));
                    curandSafeCall(curandGenerateNormal(gen, m_temp, size_temp, (float)mean, (float)sdev));
                    cudaSafeCall(cudaMemcpy(m->data, m_temp, m->nbytes, cudaMemcpyDefault));
                    cudaSafeCall(cudaFree(m_temp));
                }
                else {
                    curandSafeCall(curandGenerateNormal(gen, (float *)m->data, m->size, (float)mean, (float)sdev));
                }
                break;
            default:
                PyErr_SetString(PyExc_TypeError, "invalid data type for curand generation");
                return 0;
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, "invalid vector/matrix type for curand generation");
        return 0;
    }

    // return None
    Py_RETURN_NONE;
}

// uniform distribution generation [0, 1)
const char * const pyre::extensions::cuda::curand::uniform__name__ = "curand_uniform";
const char * const pyre::extensions::cuda::curand::uniform__doc__ = "curand generate uniform distribution samples for vector/matrix";
PyObject *
pyre::extensions::cuda::curand::
uniform(PyObject *, PyObject *args)
{
    // allocate storage for the arguments
    PyObject * genCapsule; // curand generator capsule
    PyObject * dataCapsule; // cuda vector or matrix
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "O!O!:curand_uniform", &PyCapsule_Type, &genCapsule, &PyCapsule_Type, &dataCapsule)) {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for curand_uniform");
        return nullptr;
    }
    // check curand generator capsule
    if (!PyCapsule_IsValid(genCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid curand generator capsule");
        return 0;
    }
    // get the generator
    curandGenerator_t gen =
        static_cast<curandGenerator_t>(PyCapsule_GetPointer(genCapsule, capsule_t));

    // check the data capsule type
    if (PyCapsule_IsValid(dataCapsule, pyre::extensions::cuda::vector::capsule_t)) {
        // get the vector
        cuda_vector * v = static_cast<cuda_vector *>(PyCapsule_GetPointer(dataCapsule, pyre::extensions::cuda::vector::capsule_t));
        switch(v->dtype) {
            case PYCUDA_DOUBLE:
                curandSafeCall(curandGenerateUniformDouble(gen, (double *)v->data, v->size));
                break;
            case PYCUDA_FLOAT:
                curandSafeCall(curandGenerateUniform(gen, (float *)v->data, v->size));
                break;
            default:
                PyErr_SetString(PyExc_TypeError, "invalid data type for curand generation");
                return 0;
        }
    }
    else if (PyCapsule_IsValid(dataCapsule, pyre::extensions::cuda::matrix::capsule_t)) {
        // get the matrix
        cuda_matrix * m = static_cast<cuda_matrix *>(PyCapsule_GetPointer(dataCapsule, pyre::extensions::cuda::matrix::capsule_t));
        // check the data type of matrix
        switch(m->dtype) {
            case PYCUDA_DOUBLE:
                curandSafeCall(curandGenerateUniformDouble(gen, (double *)m->data, m->size));
                break;
            case PYCUDA_FLOAT:
                curandSafeCall(curandGenerateUniform(gen, (float *)m->data, m->size));
                break;
            default:
                PyErr_SetString(PyExc_TypeError, "invalid data type for curand generation");
                return 0;
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, "invalid vector/matrix type for curand generation");
        return 0;
    }

    // return None
    Py_RETURN_NONE;
}

//end of file
