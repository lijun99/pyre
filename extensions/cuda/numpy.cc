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
#include <iostream>
#include <sstream>

// my declarations
#include "capsules.h"
#include "vector.h"
#include "matrix.h"
#include "numpy.h"
// local support
#include "exceptions.h"
#include "dtypes.h"

// access to cudalib definitions
#include <pyre/cuda.h>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// copy to a numpy array
const char * const pyre::extensions::cuda::vector::tonumpy__name__ = "vector_tonumpy";
const char * const pyre::extensions::cuda::vector::tonumpy__doc__ = "return a numpy array of cuda vector";

PyObject *
pyre::extensions::cuda::vector::tonumpy(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyArrayObject *dst;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "OO!:vector_tonumpy",
                                  &dst, &PyCapsule_Type, &self);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(self, pyre::extensions::cuda::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    cuda_vector * v = static_cast<cuda_vector *>(PyCapsule_GetPointer(self, pyre::extensions::cuda::vector::capsule_t));

    // copy the data
    cudaSafeCall(cudaMemcpy(PyArray_DATA(dst), v->data, v->nbytes, cudaMemcpyDefault));
    // cudaSafeCall(cudaDeviceSynchronize());

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// from numpy array
const char * const pyre::extensions::cuda::vector::fromnumpy__name__ = "vector_fromnumpy";
const char * const pyre::extensions::cuda::vector::fromnumpy__doc__ = "copy a numpy array to cuda vector";

PyObject *
pyre::extensions::cuda::vector::fromnumpy(PyObject *, PyObject * args) {
    // the arguments
    PyArrayObject * src; // numpy
    PyObject * dstObj; //cuda_vector
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O:vector_fromnumpy",
                                  &PyCapsule_Type, &dstObj, &src);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(dstObj, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    cuda_vector * dst = static_cast<cuda_vector *>(PyCapsule_GetPointer(dstObj, capsule_t));

    // perform copy
    cudaSafeCall(cudaMemcpy(dst->data, PyArray_DATA(src), dst->nbytes, cudaMemcpyDefault));
    
    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// to numpy array
const char * const pyre::extensions::cuda::matrix::tonumpy__name__ = "matrix_tonumpy";
const char * const pyre::extensions::cuda::matrix::tonumpy__doc__ = "return a numpy array of matrix";

PyObject *
pyre::extensions::cuda::matrix::tonumpy(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyArrayObject * dst;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "OO!:matrix_tonumpy",
                                  &dst, &PyCapsule_Type, &self);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(self, pyre::extensions::cuda::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the matrix
    cuda_matrix * m = static_cast<cuda_matrix *>(PyCapsule_GetPointer(self, pyre::extensions::cuda::matrix::capsule_t));

    // copy the data
    cudaSafeCall(cudaMemcpy(PyArray_DATA(dst), m->data, m->nbytes, cudaMemcpyDefault));
    //cudaSafeCall(cudaDeviceSynchronize());
    
    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// from numpy array
const char * const pyre::extensions::cuda::matrix::fromnumpy__name__ = "matrix_fromnumpy";
const char * const pyre::extensions::cuda::matrix::fromnumpy__doc__ = "copy a numpy array to cuda matrix";

PyObject *
pyre::extensions::cuda::matrix::fromnumpy(PyObject *, PyObject * args) {
    // the arguments
    PyArrayObject * src; // numpy
    PyObject * dstObj; //cuda_matrix
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O:matrix_fromnumpy",
                                  &PyCapsule_Type, &dstObj, &src);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(dstObj, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the two vectors
    cuda_matrix * dst = static_cast<cuda_matrix *>(PyCapsule_GetPointer(dstObj, capsule_t));

    // perform copy
    cudaSafeCall(cudaMemcpy(dst->data, PyArray_DATA(src), dst->nbytes, cudaMemcpyDefault));
    
    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// end of file
