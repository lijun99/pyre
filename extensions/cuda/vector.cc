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
// local support
#include "exceptions.h"
#include "dtypes.h"

// access to cudalib definitions
#include <pyre/cuda.h>

// allocate a gpu vector
const char * const pyre::extensions::cuda::vector::alloc__name__ = "vector_alloc";
const char * const pyre::extensions::cuda::vector::alloc__doc__ = "allocate a vector on gpu";

PyObject *
pyre::extensions::cuda::vector::
alloc(PyObject *, PyObject *args)
{
    // allocate storage for the arguments
    size_t size;
    int dtype;
    size_t nbytes;
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "kki:alloc", &size, &nbytes, &dtype)) {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for cuda.vector.alloc");
        return nullptr;
    }

    // declare a cuda vector
    cuda_vector * cvector= new cuda_vector{size, nullptr, nbytes, dtype};

    // allocate data on device
    cudaSafeCall(cudaMalloc((void **)&(cvector->data), nbytes));

    // return as a capsule
    return PyCapsule_New(cvector, capsule_t, free);
}

// (force) deallocate a gpu vector, only when python garbage collector failed to release gpu memory
const char * const pyre::extensions::cuda::vector::dealloc__name__ = "vector_dealloc";
const char * const pyre::extensions::cuda::vector::dealloc__doc__ = "deallocate a vector on gpu";

PyObject *
pyre::extensions::cuda::vector::
dealloc(PyObject *, PyObject * args)
{
    PyObject * capsule;
    int status = PyArg_ParseTuple(args, "O!:vector_dealloc", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) {
        return 0;
    }

    // get the vector
    cuda_vector * cvector =
        static_cast<cuda_vector *>(PyCapsule_GetPointer(capsule, capsule_t));

    // deallocate
    if(cvector->data != nullptr) {
       cudaSafeCall(cudaFree(cvector->data));
       cvector->data = nullptr;
    }
    Py_RETURN_NONE;
}

// destructors
void
pyre::extensions::cuda::vector::
free(PyObject * capsule)
{
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) return;
    // get the vector
    cuda_vector * cvector =
        static_cast<cuda_vector *>(PyCapsule_GetPointer(capsule, capsule_t));

    // deallocate
    if(cvector->data != nullptr)
       cudaSafeCall(cudaFree(cvector->data));
       cvector->data = nullptr;
    // and return
    return;
}


// convert PyCapsule object to c cuda_vector object
// to be used with PyArg_Parse ("O&", &converter, &Cobj)
int
pyre::extensions::cuda::vector::
converter(PyObject * capsule, cuda_vector **v)
{
    *v =  static_cast<cuda_vector *>(PyCapsule_GetPointer(capsule, capsule_t));
    if (v != NULL)
        return 1;
    else
        return 0;
}


// initialization
const char * const pyre::extensions::cuda::vector::zero__name__ = "vector_zero";
const char * const pyre::extensions::cuda::vector::zero__doc__ = "zero out the elements of a vector";
PyObject *
pyre::extensions::cuda::vector::
zero(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:vector_zero", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    cuda_vector * v = static_cast<cuda_vector *>(PyCapsule_GetPointer(capsule, capsule_t));

    // fill it out
    cudaSafeCall(cudaMemset(v->data, 0, v->nbytes));

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// a test routine for using O& parser with converter
// provides the same function as fill below
const char * const pyre::extensions::cuda::vector::filla__name__ = "vector_filla";
const char * const pyre::extensions::cuda::vector::filla__doc__ = "fill out a vector with a given value";
PyObject *
pyre::extensions::cuda::vector::
filla(PyObject *, PyObject * args) {
    // the arguments
    double value;
    cuda_vector *v= 0;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O&d:vector_filla", &converter, &v, &value);
    // if something went wrong
    if (!status) return 0;

    // fill it out
    switch(v->dtype) {
    case PYCUDA_DOUBLE: //double
        cudalib::elementwise::fill<double>((double *)v->data, v->size, value);
        break;
    case PYCUDA_FLOAT: //single
        cudalib::elementwise::fill<float>((float *)v->data, v->size, (float)value);
        break;
    case PYCUDA_INT: //int32
        cudalib::elementwise::fill<int>((int *)v->data, v->size, (int)value);
        break;
    case PYCUDA_LONG: //int64
        cudalib::elementwise::fill<long>((long *)v->data, v->size, (long)value);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "types other than float/double/int/long are not implemented yet");
        return 0;
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const pyre::extensions::cuda::vector::fill__name__ = "vector_fill";
const char * const pyre::extensions::cuda::vector::fill__doc__ = "fill out a vector with a given value";
PyObject *
pyre::extensions::cuda::vector::
fill(PyObject *, PyObject * args) {
    // the arguments
    double value;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!d:vector_fill", &PyCapsule_Type, &capsule, &value);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    cuda_vector * v = static_cast<cuda_vector *>(PyCapsule_GetPointer(capsule, capsule_t));

    // fill it out
    switch(v->dtype) {
    case PYCUDA_DOUBLE: //double
        cudalib::elementwise::fill<double>((double *)v->data, v->size, value);
        break;
    case PYCUDA_FLOAT: //single
        cudalib::elementwise::fill<float>((float *)v->data, v->size, (float)value);
        break;
    case PYCUDA_INT: //int32
        cudalib::elementwise::fill<int>((int *)v->data, v->size, (int)value);
        break;
    case PYCUDA_LONG: //int64
        cudalib::elementwise::fill<long>((long *)v->data, v->size, (long)value);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "types other than float/double/int/long are not implemented yet");
        return 0;
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// copy to another vector
const char * const pyre::extensions::cuda::vector::copy__name__ = "vector_copy";
const char * const pyre::extensions::cuda::vector::copy__doc__ = "copy vector to another (cuda)vector";

PyObject *
pyre::extensions::cuda::vector::
copy(PyObject *, PyObject * args) {
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
    if (!PyCapsule_IsValid(srcObj, capsule_t) || !PyCapsule_IsValid(dstObj, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    cuda_vector * src = static_cast<cuda_vector *>(PyCapsule_GetPointer(srcObj, capsule_t));
    cuda_vector * dst = static_cast<cuda_vector *>(PyCapsule_GetPointer(dstObj, capsule_t));

    // perform copy
    cudaSafeCall(cudaMemcpy(dst->data, src->data, src->nbytes, cudaMemcpyDefault));

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// add a+=b
const char * const pyre::extensions::cuda::vector::iadd__name__ = "vector_iadd";
const char * const pyre::extensions::cuda::vector::iadd__doc__ = "in-place addition of two vectors";

PyObject *
pyre::extensions::cuda::vector::iadd(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyObject * other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:vector_iadd",
                                  &PyCapsule_Type, &self, &PyCapsule_Type, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t) || !PyCapsule_IsValid(other, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    cuda_vector * v1 = static_cast<cuda_vector *>(PyCapsule_GetPointer(self, capsule_t));
    cuda_vector * v2 = static_cast<cuda_vector *>(PyCapsule_GetPointer(other, capsule_t));

    // perform the addition
    switch(v1->dtype) {
    case PYCUDA_DOUBLE: //double
        cudalib::elementwise::iadd<double>((double *)v1->data, (double *)v2->data, v1->size);
        break;
    case PYCUDA_FLOAT: //single
        cudalib::elementwise::iadd<float>((float *)v1->data, (float *)v2->data, v1->size);
        break;
    case PYCUDA_INT: //int
        cudalib::elementwise::iadd<int>((int *)v1->data, (int *)v2->data, v1->size);
        break;
    case PYCUDA_LONG: //single
        cudalib::elementwise::iadd<long>((long *)v1->data, (long *)v2->data, v1->size);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "types other than float/double are not implemented yet");
        return 0;
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const pyre::extensions::cuda::vector::isub__name__ = "vector_isub";
const char * const pyre::extensions::cuda::vector::isub__doc__ = "in-place subtraction of two vectors";

PyObject *
pyre::extensions::cuda::vector::isub(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyObject * other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:vector_isub",
                                  &PyCapsule_Type, &self, &PyCapsule_Type, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t) || !PyCapsule_IsValid(other, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    cuda_vector * v1 = static_cast<cuda_vector *>(PyCapsule_GetPointer(self, capsule_t));
    cuda_vector * v2 = static_cast<cuda_vector *>(PyCapsule_GetPointer(other, capsule_t));

    // perform the subtraction
    switch(v1->dtype) {
    case PYCUDA_DOUBLE: //double
        cudalib::elementwise::isub<double>((double *)v1->data, (double *)v2->data, v1->size);
        break;
    case PYCUDA_FLOAT: //single
        cudalib::elementwise::isub<float>((float *)v1->data, (float *)v2->data, v1->size);
        break;
    case PYCUDA_INT: //int
        cudalib::elementwise::isub<int>((int *)v1->data, (int *)v2->data, v1->size);
        break;
    case PYCUDA_LONG: //long
        cudalib::elementwise::isub<long>((long *)v1->data, (long *)v2->data, v1->size);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "types other than float/double/int/long are not implemented yet");
        return 0;
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

const char * const pyre::extensions::cuda::vector::imul__name__ = "vector_imul";
const char * const pyre::extensions::cuda::vector::imul__doc__ = "in-place multiplication of two vectors";

PyObject *
pyre::extensions::cuda::vector::imul(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyObject * other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:vector_imul",
                                  &PyCapsule_Type, &self, &PyCapsule_Type, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t) || !PyCapsule_IsValid(other, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    cuda_vector * v1 = static_cast<cuda_vector *>(PyCapsule_GetPointer(self, capsule_t));
    cuda_vector * v2 = static_cast<cuda_vector *>(PyCapsule_GetPointer(other, capsule_t));

    // perform the subtraction
    switch(v1->dtype) {
    case PYCUDA_DOUBLE: //double
        cudalib::elementwise::imul<double>((double *)v1->data, (double *)v2->data, v1->size);
        break;
    case PYCUDA_FLOAT: //single
        cudalib::elementwise::imul<float>((float *)v1->data, (float *)v2->data, v1->size);
        break;
    case PYCUDA_INT: //int
        cudalib::elementwise::imul<int>((int *)v1->data, (int *)v2->data, v1->size);
        break;
    case PYCUDA_LONG: //long
        cudalib::elementwise::imul<long>((long *)v1->data, (long *)v2->data, v1->size);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "types other than float/double/int/long are not implemented yet");
        return 0;
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// add a+=b
const char * const pyre::extensions::cuda::vector::iadd_scalar__name__ = "vector_iadd_scalar";
const char * const pyre::extensions::cuda::vector::iadd_scalar__doc__ = "add a scalar to vector";

PyObject *
pyre::extensions::cuda::vector::iadd_scalar(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    double other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!d:vector_iadd_scalar",
                                  &PyCapsule_Type, &self, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    cuda_vector * v1 = static_cast<cuda_vector *>(PyCapsule_GetPointer(self, capsule_t));

    // perform the subtraction
    switch(v1->dtype) {
    case PYCUDA_DOUBLE: //double
        cudalib::elementwise::iadd_scalar<double>((double *)v1->data, other, v1->size);
        break;
    case PYCUDA_FLOAT: //single
        cudalib::elementwise::iadd_scalar<float>((float *)v1->data, (float)other, v1->size);
        break;
    case PYCUDA_INT: // int32
        cudalib::elementwise::iadd_scalar<int>((int *)v1->data, (int)other, v1->size);
        break;
    case PYCUDA_LONG: // int64
        cudalib::elementwise::iadd_scalar<long>((long *)v1->data, (long)other, v1->size);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "types other than float/double are not implemented yet");
        return 0;
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// imul a1*=a2 (a2 scalar)
const char * const pyre::extensions::cuda::vector::imul_scalar__name__ = "vector_imul_scalar";
const char * const pyre::extensions::cuda::vector::imul_scalar__doc__ = "in-place subtraction of two vectors";

PyObject *
pyre::extensions::cuda::vector::imul_scalar(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    double other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!d:vector_imul_scalar",
                                  &PyCapsule_Type, &self, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the two vectors
    cuda_vector * v1 = static_cast<cuda_vector *>(PyCapsule_GetPointer(self, capsule_t));

    // perform the subtraction
    switch(v1->dtype) {
    case PYCUDA_DOUBLE: //double
        cudalib::elementwise::imul_scalar<double>((double *)v1->data, other, v1->size);
        break;
    case PYCUDA_FLOAT: //single
        cudalib::elementwise::imul_scalar<float>((float *)v1->data, (float)other, v1->size);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "types other than float/double are not implemented yet");
        return 0;
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}
// end of file
