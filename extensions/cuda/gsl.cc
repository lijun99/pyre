// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

#include <portinfo>
#include <Python.h>
#include <pyre/journal.h>

// my declarations
#include "capsules.h"
#include "gsl.h"
#include "vector.h"
#include "matrix.h"
// local support
#include "exceptions.h"
#include "dtypes.h"
// so I can build reasonable error messages
#include <sstream>

// access to cudalib definitions
#include <pyre/cuda.h>

// access the gsl vector
#include <pyre/gsl/capsules.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include <iostream>

// copy to gsl(host)
const char * const pyre::extensions::cuda::vector::togsl__name__ = "vector_togsl";
const char * const pyre::extensions::cuda::vector::togsl__doc__ = "copy vector to gsl(host)";

PyObject *
pyre::extensions::cuda::vector::togsl(PyObject *, PyObject * args) {
    // the arguments
    PyObject * srcObj;
    PyObject * dstObj;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:vector_togsl",
                                  &PyCapsule_Type, &dstObj, &PyCapsule_Type, &srcObj);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(srcObj, capsule_t) || !PyCapsule_IsValid(dstObj, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    cuda_vector * src = static_cast<cuda_vector *>(PyCapsule_GetPointer(srcObj, capsule_t));
    gsl_vector * dst = static_cast<gsl_vector *>(PyCapsule_GetPointer(dstObj, gsl::vector::capsule_t));

    // perform copy
    // gsl_vector is double precision
    switch(src->dtype) {
        case PYCUDA_DOUBLE:
        {
            //double to double, direct memcpy
            cudaSafeCall(cudaMemcpy(dst->data, src->data, src->nbytes, cudaMemcpyDeviceToHost));
            break;
        }
        case PYCUDA_FLOAT:
        {
            //float to double, use a double array on gpu as conversion buffer
            double * d_double;
            cudaSafeCall(cudaMalloc((void **)&d_double, src->size*sizeof(double)));
            // convert float to double on gpu
            cudalib::elementwise::conversion<double, float>(d_double, (float *)src->data, src->size);
            cudaSafeCall(cudaMemcpy(dst->data, d_double, src->size*sizeof(double), cudaMemcpyDefault));
            cudaSafeCall(cudaFree(d_double));
            break;
        }
        default:
        {
            PyErr_SetString(PyExc_TypeError, "Types other than float/double are not supported");
            return 0;
        }
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// copy from gsl(host)
const char * const pyre::extensions::cuda::vector::fromgsl__name__ = "vector_fromgsl";
const char * const pyre::extensions::cuda::vector::fromgsl__doc__ = "copy vector from gsl(host)";

PyObject *
pyre::extensions::cuda::vector::fromgsl(PyObject *, PyObject * args) {
    // the arguments
    PyObject * srcObj; //gsl_vector
    PyObject * dstObj; //cuda_vector
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:vector_fromgsl",
                                  &PyCapsule_Type, &dstObj, &PyCapsule_Type, &srcObj);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(dstObj, capsule_t) || !PyCapsule_IsValid(srcObj, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    cuda_vector * dst = static_cast<cuda_vector *>(PyCapsule_GetPointer(dstObj, capsule_t));
    gsl_vector * src = static_cast<gsl_vector *>(PyCapsule_GetPointer(srcObj, gsl::vector::capsule_t));

    // perform copy
    switch(dst->dtype) {
        case PYCUDA_DOUBLE:
        {
            //double to double
            cudaSafeCall(cudaMemcpy(dst->data, src->data, dst->nbytes, cudaMemcpyDefault));
            break;
        }
        case PYCUDA_FLOAT:
        {
            //float to double
            // create a temporary array on gpu with float type
            double * d_double;
            size_t nbytes = src->size*sizeof(double);
            cudaSafeCall(cudaMalloc((void **)&d_double, nbytes));
            // copy data from host to device(stored in temp) 
            cudaSafeCall(cudaMemcpy(d_double, src->data, nbytes, cudaMemcpyDefault));
            // convert double to single precision on gpu
            cudalib::elementwise::conversion<float, double>((float *)dst->data, d_double, dst->size);
            cudaSafeCall(cudaFree(d_double));
            break;
        }
        default:
            PyErr_SetString(PyExc_TypeError, "Types other than float/double are not supported");
            return 0;        
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// copy to gsl(host)
const char * const pyre::extensions::cuda::matrix::togsl__name__ = "matrix_togsl";
const char * const pyre::extensions::cuda::matrix::togsl__doc__ = "copy matrix to gsl(host)";

PyObject *
pyre::extensions::cuda::matrix::togsl(PyObject *, PyObject * args) {
    // the arguments
    PyObject * srcObj; //cuda_matrix
    PyObject * dstObj; //gsl_matrix 
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:matrix_togsl",
                                  &PyCapsule_Type, &dstObj, &PyCapsule_Type, &srcObj);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(srcObj, capsule_t) || !PyCapsule_IsValid(dstObj, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the two matrixs
    cuda_matrix * src = static_cast<cuda_matrix *>(PyCapsule_GetPointer(srcObj, capsule_t));
    gsl_matrix * dst = static_cast<gsl_matrix *>(PyCapsule_GetPointer(dstObj, gsl::matrix::capsule_t));

    // perform copy
    // note gsl_matrix is double precision
    switch(src->dtype) {
    case PYCUDA_DOUBLE:
        //double to double, direct memcpy
        cudaSafeCall(cudaMemcpy(dst->data, src->data, src->nbytes, cudaMemcpyDeviceToHost));
        break;
    case PYCUDA_FLOAT: {
        //float to double, use a double array on gpu as conversion buffer
        double * d_array;
        size_t d_nbytes = src->size*sizeof(double);
        cudaSafeCall(cudaMalloc((void **)&d_array, d_nbytes));
        // convert float to double on gpu
        cudalib::elementwise::conversion<double, float>(d_array, (float *)src->data, src->size);
        cudaSafeCall(cudaMemcpy(dst->data, d_array, d_nbytes, cudaMemcpyDefault));
        cudaSafeCall(cudaFree(d_array));
        break;}
    default:
        PyErr_SetString(PyExc_TypeError, "Types other than float/double are not supported");
        return 0;        
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// copy from gsl(host)
const char * const pyre::extensions::cuda::matrix::fromgsl__name__ = "matrix_fromgsl";
const char * const pyre::extensions::cuda::matrix::fromgsl__doc__ = "copy matrix from gsl(host)";

PyObject *
pyre::extensions::cuda::matrix::fromgsl(PyObject *, PyObject * args) {
    // the arguments
    PyObject * srcObj; //gsl_matrix
    PyObject * dstObj; //cuda_matrix
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:matrix_fromgsl",
                                  &PyCapsule_Type, &dstObj, &PyCapsule_Type, &srcObj);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(dstObj, capsule_t) || !PyCapsule_IsValid(srcObj, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the two matrixs
    cuda_matrix * dst = static_cast<cuda_matrix *>(PyCapsule_GetPointer(dstObj, capsule_t));
    gsl_matrix * src = static_cast<gsl_matrix *>(PyCapsule_GetPointer(srcObj, gsl::matrix::capsule_t));

    // perform copy
    switch(dst->dtype) {
    case PYCUDA_DOUBLE:
        // double to double
        cudaSafeCall(cudaMemcpy(dst->data, src->data, dst->nbytes, cudaMemcpyDefault));
        break;
    case PYCUDA_FLOAT:
        {
        //float to double
        // create a temporary array on gpu with float type
        double * d_double;
        size_t size = src->size1 * src->size2; 
        size_t d_nbytes = size*sizeof(double);
        cudaSafeCall(cudaMalloc((void **)&d_double, d_nbytes));
        // copy data from host to device(stored in temp) 
        cudaSafeCall(cudaMemcpy(d_double, src->data, d_nbytes, cudaMemcpyDefault));
        // convert double to single precision on gpu
        cudalib::elementwise::conversion<float, double>((float *)dst->data, d_double, size);
        cudaSafeCall(cudaFree(d_double));
        break; }
    default:
        PyErr_SetString(PyExc_TypeError, "Types other than float/double are not supported");
        return 0;    
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}
