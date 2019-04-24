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
#include "matrix.h"
// local support
#include "exceptions.h"
#include "dtypes.h"

// access to cudalib definitions
#include <pyre/cuda.h>


// allocate a gpu matrix
const char * const pyre::extensions::cuda::matrix::alloc__name__ = "matrix_alloc";
const char * const pyre::extensions::cuda::matrix::alloc__doc__ = "allocate a matrix on gpu";
                  
PyObject *
pyre::extensions::cuda::matrix::
alloc(PyObject *, PyObject *args)
{
    // allocate storage for the arguments
    size_t size1, size2;
    int dtype;
    size_t nbytes;
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "(kk)ki:alloc", &size1, &size2, &nbytes, &dtype)) {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for cuda.matrix_alloc"); 
        return nullptr;
    }
    
    // declare a cuda matrix
    cuda_matrix *cmatrix= new cuda_matrix{size1, size2, size1*size2, nullptr, nbytes, dtype};

    // allocate data on device
    cudaSafeCall(cudaMalloc((void **)&(cmatrix->data), nbytes));

    // return as a capsule
    return PyCapsule_New(cmatrix, capsule_t, free);
}

// destructors
void
pyre::extensions::cuda::matrix::
free(PyObject * capsule)
{
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, pyre::extensions::cuda::matrix::capsule_t)) return;
    // get the matrix
    cuda_matrix * cmatrix =
        static_cast<cuda_matrix *>(PyCapsule_GetPointer(capsule, pyre::extensions::cuda::matrix::capsule_t));
    
    // deallocate
    if(!cmatrix->data)
       cudaSafeCall(cudaFree(cmatrix->data)); 
    // and return
    return;
}


// initialization
const char * const pyre::extensions::cuda::matrix::zero__name__ = "matrix_zero";
const char * const pyre::extensions::cuda::matrix::zero__doc__ = "zero out the elements of a matrix";
PyObject *
pyre::extensions::cuda::matrix::
zero(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:matrix_zero", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the matrix
    cuda_matrix * m = static_cast<cuda_matrix *>(PyCapsule_GetPointer(capsule, capsule_t));

    // fill it out
    cudaSafeCall(cudaMemset(m->data, 0, m->nbytes));

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const pyre::extensions::cuda::matrix::fill__name__ = "matrix_fill";
const char * const pyre::extensions::cuda::matrix::fill__doc__ = "fill out a matrix with a given value";
PyObject *
pyre::extensions::cuda::matrix::
fill(PyObject *, PyObject * args) {
    // the arguments
    double value;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!d:matrix_fill", &PyCapsule_Type, &capsule, &value);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the matrix
    cuda_matrix * m = static_cast<cuda_matrix *>(PyCapsule_GetPointer(capsule, capsule_t));

    // fill it out
    switch(m->dtype) {
    case PYCUDA_DOUBLE: //double
        cudalib::elementwise::fill<double>((double *)m->data, m->size, value);
        break;
    case PYCUDA_FLOAT: //single
        cudalib::elementwise::fill<float>((float *)m->data, m->size, (float)value);
        break;
    case PYCUDA_INT: //int32
        cudalib::elementwise::fill<int>((int *)m->data, m->size, (int)value);
        break;
    case PYCUDA_LONG: //single
        cudalib::elementwise::fill<long>((long *)m->data, m->size, (long)value);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "types other than float/double/int/long are not implenmented yet");
        return 0;
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// iadd a+=b
const char * const pyre::extensions::cuda::matrix::iadd__name__ = "matrix_iadd";
const char * const pyre::extensions::cuda::matrix::iadd__doc__ = "in-place addition of two matrixs";

PyObject *
pyre::extensions::cuda::matrix::iadd(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyObject * other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:matrix_iadd",
                                  &PyCapsule_Type, &self, &PyCapsule_Type, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t) || !PyCapsule_IsValid(other, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the two matrixs
    cuda_matrix * m1 = static_cast<cuda_matrix *>(PyCapsule_GetPointer(self, capsule_t));
    cuda_matrix * m2 = static_cast<cuda_matrix *>(PyCapsule_GetPointer(other, capsule_t));

    size_t size = m1->size;

    // perform the addition
    switch(m1->dtype) {
    case PYCUDA_DOUBLE: //double
        cudalib::elementwise::iadd<double>((double *)m1->data, (double *)m2->data, size);
        break;
    case PYCUDA_FLOAT: //single
        cudalib::elementwise::iadd<float>((float *)m1->data, (float *)m2->data, size);
        break;
    case PYCUDA_INT: // int32
        cudalib::elementwise::iadd<int>((int *)m1->data, (int *)m2->data, size);
        break;
    case PYCUDA_LONG: // int64
        cudalib::elementwise::iadd<long>((long *)m1->data, (long *)m2->data, size);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "types other than float/double/int/long are not implenmented yet");
        return 0;
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


const char * const pyre::extensions::cuda::matrix::isub__name__ = "matrix_isub";
const char * const pyre::extensions::cuda::matrix::isub__doc__ = "in-place subtraction of two matrixs";

PyObject *
pyre::extensions::cuda::matrix::isub(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    PyObject * other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:matrix_isub",
                                  &PyCapsule_Type, &self, &PyCapsule_Type, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t) || !PyCapsule_IsValid(other, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the two matrixs
    cuda_matrix * m1 = static_cast<cuda_matrix *>(PyCapsule_GetPointer(self, capsule_t));
    cuda_matrix * m2 = static_cast<cuda_matrix *>(PyCapsule_GetPointer(other, capsule_t));

    size_t size = m1->size;

    // perform the subtraction
    switch(m1->dtype) {
    case PYCUDA_DOUBLE: //double
        cudalib::elementwise::isub<double>((double *)m1->data, (double *)m2->data, size);
        break;
    case PYCUDA_FLOAT: //single
        cudalib::elementwise::isub<float>((float *)m1->data, (float *)m2->data, size);
        break;
    case PYCUDA_INT: // int32
        cudalib::elementwise::isub<int>((int *)m1->data, (int *)m2->data, size);
        break;
    case PYCUDA_LONG: // int64
        cudalib::elementwise::isub<long>((long *)m1->data, (long *)m2->data, size);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "types other than float/double/int/long are not implenmented yet");
        return 0;
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// a1*=a2 (scalar)
const char * const pyre::extensions::cuda::matrix::imul__name__ = "matrix_imul";
const char * const pyre::extensions::cuda::matrix::imul__doc__ = "matrix scale";

PyObject *
pyre::extensions::cuda::matrix::imul(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    double other;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!d:matrix_imul",
                                  &PyCapsule_Type, &self, &other);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(self, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the two matrixs
    cuda_matrix * m1 = static_cast<cuda_matrix *>(PyCapsule_GetPointer(self, capsule_t));

    size_t size = m1->size;

    // perform the subtraction
    switch(m1->dtype) {
    case PYCUDA_DOUBLE: //double
        cudalib::elementwise::imul<double>((double *)m1->data, other, size);
        break;
    case PYCUDA_FLOAT: //single
        cudalib::elementwise::imul<float>((float *)m1->data, (float)other, size);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "types other than float/double are not implenmented yet");
        return 0;
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// copy to another matrix
const char * const pyre::extensions::cuda::matrix::copy__name__ = "matrix_copy";
const char * const pyre::extensions::cuda::matrix::copy__doc__ = "copy matrix to another (cuda)matrix";

PyObject *
pyre::extensions::cuda::matrix::
copy(PyObject *, PyObject * args) {
    // the arguments
    PyObject * srcObj;
    PyObject * dstObj;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:matrix_copy",
                                  &PyCapsule_Type, &dstObj, &PyCapsule_Type, &srcObj);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(srcObj, capsule_t) || !PyCapsule_IsValid(dstObj, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the two matrixs
    cuda_matrix * src = static_cast<cuda_matrix *>(PyCapsule_GetPointer(srcObj, capsule_t));
    cuda_matrix * dst = static_cast<cuda_matrix *>(PyCapsule_GetPointer(dstObj, capsule_t));

    // perform copy
    cudaSafeCall(cudaMemcpy(dst->data, src->data, src->nbytes, cudaMemcpyDefault));

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// copytile : submatrix or insertion
const char * const pyre::extensions::cuda::matrix::copytile__name__ = "matrix_copytile";
const char * const pyre::extensions::cuda::matrix::copytile__doc__ = "copy a copytile/submatrix to another matrix";

PyObject *
pyre::extensions::cuda::matrix::
copytile(PyObject *, PyObject * args) {
    // the arguments
    PyObject * srcObj;
    size_t srowStart, scolStart;
    PyObject * dstObj;
    size_t drowStart, dcolStart;
    size_t rows, cols;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!(kk)O!(kk)(kk):matrix_copytile",
                                  &PyCapsule_Type, &dstObj, &drowStart, &dcolStart,
                                  &PyCapsule_Type, &srcObj, &srowStart, &scolStart,
                                  &rows, &cols);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(srcObj, capsule_t) || !PyCapsule_IsValid(dstObj, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the two matrixs
    cuda_matrix * src = static_cast<cuda_matrix *>(PyCapsule_GetPointer(srcObj, capsule_t));
    cuda_matrix * dst = static_cast<cuda_matrix *>(PyCapsule_GetPointer(dstObj, capsule_t));

    // perform copy of submatrix
/*
template<typename Tout, typename Tin>
void cudalib::matrix::
copy_tile(Tout* const odata,  const size_t ldo, 
                const size_t omstart, const size_t onstart, // starting position of odata
                const Tin* const idata, const size_t ldi,
                const size_t imstart, const size_t instart, // starting position of idata
                const size_t m, const size_t n, // tile to be copied
                cudaStream_t stream)
*/
    switch(src->dtype) {
    case PYCUDA_FLOAT:
        cudalib::matrix::copy_tile<float, float>((float * const)dst->data, dst->size2,
                                    drowStart, dcolStart,
                                    (const float * const)src->data, src->size2,
                                    srowStart, scolStart,
                                    rows, cols);
        break;
    case PYCUDA_DOUBLE:
        cudalib::matrix::copy_tile<double, double>((double * const)dst->data, dst->size2,
                                    drowStart, dcolStart,
                                    (const double * const)src->data, src->size2,
                                    srowStart, scolStart,
                                    rows, cols);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "data types other than float/double are not supported yet");
        return 0;
    } //end of switch
    
    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// copycols : copy selected cols with indices
const char * const pyre::extensions::cuda::matrix::copycols__name__ = "matrix_copycols";
const char * const pyre::extensions::cuda::matrix::copycols__doc__ = "copy a copycols/submatrix to another matrix";

PyObject *
pyre::extensions::cuda::matrix::
copycols(PyObject *, PyObject * args) {
    // the arguments
    PyObject * srcObj;
    PyObject * dstObj;
    size_t rows, cols;
    PyObject * idxObj;    
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!(kk)O!:matrix_copycols",
                                  &PyCapsule_Type, &dstObj,
                                  &PyCapsule_Type, &srcObj,
                                  &rows, &cols,
                                  &PyCapsule_Type, &idxObj
                                  );
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(srcObj, capsule_t) || !PyCapsule_IsValid(dstObj, capsule_t)
        || !PyCapsule_IsValid(idxObj, pyre::extensions::cuda::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the two matrixs
    cuda_matrix * src = static_cast<cuda_matrix *>(PyCapsule_GetPointer(srcObj, capsule_t));
    cuda_matrix * dst = static_cast<cuda_matrix *>(PyCapsule_GetPointer(dstObj, capsule_t));
    cuda_vector * idx = static_cast<cuda_vector *>(PyCapsule_GetPointer(idxObj, pyre::extensions::cuda::vector::capsule_t));

    switch(src->dtype) {
    case PYCUDA_FLOAT:
        cudalib::matrix::copy_indices<float>((float * const)dst->data, dst->size2,
                                    (const float * const)src->data, src->size2,
                                    rows, cols, (const size_t *const)idx->data);
        break;
    case PYCUDA_DOUBLE:
        cudalib::matrix::copy_indices<double>((double * const)dst->data, dst->size2,
                                    (const double * const)src->data, src->size2,
                                    rows, cols, (const size_t *const)idx->data);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "data types other than float/double are not supported yet");
        return 0;
    } //end of switch
    
    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// duplicate a vector
const char * const pyre::extensions::cuda::matrix::duplicate_vector__name__ = "matrix_duplicate_vector";
const char * const pyre::extensions::cuda::matrix::duplicate_vector__doc__ = "copy duplicates of vector to another matrix";

PyObject *
pyre::extensions::cuda::matrix::
duplicate_vector(PyObject *, PyObject * args) {
    // the arguments
    PyObject * srcObj;
    PyObject * dstObj;
    size_t rows, cols, incx;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!(kk)k:matrix_duplicate_vector",
                                  &PyCapsule_Type, &dstObj,
                                  &PyCapsule_Type, &srcObj,
                                  &rows, &cols, &incx);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(srcObj, pyre::extensions::cuda::vector::capsule_t)
        || !PyCapsule_IsValid(dstObj, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix/vector capsule");
        return 0;
    }

    // get the two matrixs
    cuda_vector * src = static_cast<cuda_vector *>(PyCapsule_GetPointer(srcObj, pyre::extensions::cuda::vector::capsule_t));
    cuda_matrix * dst = static_cast<cuda_matrix *>(PyCapsule_GetPointer(dstObj, capsule_t));

    // perform copy of submatrix
    /*
    template<typename T>
    void cudalib::matrix::
    duplicate_vector(T* const odata,  const size_t ldo, 
                const T* const idata, const size_t incx,
                const size_t m, const size_t n, // tile to be copied
                cudaStream_t stream)
    */ 
    switch(src->dtype) {
    case PYCUDA_FLOAT:
        cudalib::matrix::duplicate_vector<float>((float * const)dst->data, dst->size2,
                                    (const float * const)src->data, incx,
                                    rows, cols);
        break;
    case PYCUDA_DOUBLE:
        cudalib::matrix::duplicate_vector<double>((double * const)dst->data, dst->size2,
                                    (const double * const)src->data, incx,
                                    rows, cols);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "data types other than float/double are not supported yet");
        return 0;
    } //end of switch
    
    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// transpose
const char * const pyre::extensions::cuda::matrix::transpose__name__ = "matrix_transpose";
const char * const pyre::extensions::cuda::matrix::transpose__doc__ = "copy a transpose/submatrix to another matrix";

PyObject *
pyre::extensions::cuda::matrix::
transpose(PyObject *, PyObject * args) {
    // the arguments
    PyObject * srcObj;
    PyObject * dstObj;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:matrix_transpose",
                                  &PyCapsule_Type, &dstObj, &PyCapsule_Type, &srcObj);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(srcObj, capsule_t) || !PyCapsule_IsValid(dstObj, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the two matrixs
    cuda_matrix * src = static_cast<cuda_matrix *>(PyCapsule_GetPointer(srcObj, capsule_t));
    cuda_matrix * dst = static_cast<cuda_matrix *>(PyCapsule_GetPointer(dstObj, capsule_t));

    // perform transpose
/*
        template<typename T>
        void transpose(T * const odata, const T* const idata, const size_t iM, const size_t iN, cudaStream_t stream=0);
*/
    switch(src->dtype) {
    case PYCUDA_FLOAT:
        cudalib::matrix::transpose<float>((float * const)dst->data,
                                    (const float * const)src->data, src->size1, src->size2);
        break;
    case PYCUDA_DOUBLE:
        cudalib::matrix::transpose<double>((double * const)dst->data,
                                    (const double * const)src->data, src->size1, src->size2);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "data types other than float/double are not supported yet");
        return 0;
    } //end of switch
    
    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// inverse by LU 
const char * const pyre::extensions::cuda::matrix::inverse__name__ = "matrix_inverse";
const char * const pyre::extensions::cuda::matrix::inverse__doc__ = "matrix inverse";

PyObject *
pyre::extensions::cuda::matrix::
inverse(PyObject *, PyObject * args) {
    // the arguments
    PyObject * matrixCapsule;
    PyObject * cublasCapsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:matrix_inverse",
                                  &PyCapsule_Type, &cublasCapsule, &PyCapsule_Type, &matrixCapsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(matrixCapsule, capsule_t)){ 
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }
    if (!PyCapsule_IsValid(cublasCapsule, pyre::extensions::cuda::cublas::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid cublas_handle capsule");
        return 0;
    }

    // get the two matrixs
    cuda_matrix * m = static_cast<cuda_matrix *>(PyCapsule_GetPointer(matrixCapsule, capsule_t));
    cublasHandle_t handle = static_cast<cublasHandle_t>(PyCapsule_GetPointer(cublasCapsule, pyre::extensions::cuda::cublas::capsule_t));

    const size_t n = m->size1;
    // perform inverse
    switch(m->dtype) {
    case PYCUDA_DOUBLE: //double
        cudalib::linalg::inverse_cublas<double>(handle, (double *)m->data, n);
        break;
    case PYCUDA_FLOAT: //single
        cudalib::linalg::inverse_cublas<float>(handle, (float *)m->data, n);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "wrong matrix data type");
        return 0;
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// inverse by cholesky
const char * const pyre::extensions::cuda::matrix::inverse_symm__name__ = "matrix_inverse_symm";
const char * const pyre::extensions::cuda::matrix::inverse_symm__doc__ = "matrix inverse for symmetric";

PyObject *
pyre::extensions::cuda::matrix::
inverse_symm(PyObject *, PyObject * args) {
    // the arguments
    PyObject * matrixCapsule;
    int Uplo;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!i:matrix_inverse_symm",
                                  &PyCapsule_Type, &matrixCapsule, &Uplo);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(matrixCapsule, capsule_t)){ 
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the two matrixs
    cuda_matrix * mat = static_cast<cuda_matrix *>(PyCapsule_GetPointer(matrixCapsule, capsule_t));
    cublasFillMode_t uplo = (cublasFillMode_t)Uplo;

    const size_t n = mat->size1;
    // perform inverse
    switch(mat->dtype) {
    case PYCUDA_DOUBLE: //double
        cudalib::linalg::inverse_symm_D((double *)mat->data, uplo, n);
        break;
    case PYCUDA_FLOAT: //single
        cudalib::linalg::inverse_symm_S((float *)mat->data, uplo, n);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "wrong matrix data type");
        return 0;
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// Cholesky decomposition
const char * const pyre::extensions::cuda::matrix::cholesky__name__ = "matrix_cholesky";
const char * const pyre::extensions::cuda::matrix::cholesky__doc__ = "matrix cholesky";

PyObject *
pyre::extensions::cuda::matrix::
cholesky(PyObject *, PyObject * args) {
    // the arguments
    PyObject * matrixCapsule;
    int UpLo;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!i:matrix_cholesky",
                                  &PyCapsule_Type, &matrixCapsule, &UpLo);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(matrixCapsule, capsule_t)){ 
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }
    

    // get the two matrixs
    cuda_matrix * m = static_cast<cuda_matrix *>(PyCapsule_GetPointer(matrixCapsule, capsule_t));
    cublasFillMode_t uplo = (cublasFillMode_t)UpLo;
    
    const size_t n = m->size1;
    // perform cholesky
    switch(m->dtype) {
    case PYCUDA_DOUBLE: //double
        cudalib::linalg::cholesky<double>((double *)m->data, uplo, n);
        break;
    case PYCUDA_FLOAT: //single
        cudalib::linalg::cholesky<float>((float *)m->data, uplo, n);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "wrong matrix data type");
        return 0;
    }

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// determinant of a triangular matrix i.e. product of diagonal elements
const char * const pyre::extensions::cuda::matrix::determinant_triangular__name__ = "matrix_determinant_triangular";
const char * const pyre::extensions::cuda::matrix::determinant_triangular__doc__ =
    "matrix determinant of a triangular shape";

PyObject *
pyre::extensions::cuda::matrix::
determinant_triangular(PyObject *, PyObject * args) {
    // the arguments
    PyObject * matrixCapsule;
    double det;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!:matrix_determinant_triangular",
                                  &PyCapsule_Type, &matrixCapsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(matrixCapsule, capsule_t)){ 
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }
    

    // get the two matrixs
    cuda_matrix * m = static_cast<cuda_matrix *>(PyCapsule_GetPointer(matrixCapsule, capsule_t));
    
    const size_t n = m->size1;
    // perform determinant_triangular
    switch(m->dtype) {
    case PYCUDA_DOUBLE: //double
        det= cudalib::linalg::determinant_triangular<double>((double *)m->data, n);
        break;
    case PYCUDA_FLOAT: //single
        det = (double) cudalib::linalg::determinant_triangular<float>((float *)m->data, n);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "wrong matrix data type");
        return 0;
    }

    // return 
    return PyFloat_FromDouble(det);
}

// determinant of a triangular matrix i.e. product of diagonal elements
const char * const pyre::extensions::cuda::matrix::logdet_triangular__name__ = "matrix_logdet_triangular";
const char * const pyre::extensions::cuda::matrix::logdet_triangular__doc__ =
    "matrix determinant of a triangular shape";

PyObject *
pyre::extensions::cuda::matrix::
logdet_triangular(PyObject *, PyObject * args) {
    // the arguments
    PyObject * matrixCapsule;
    double det;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!:matrix_logdet_triangular",
                                  &PyCapsule_Type, &matrixCapsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(matrixCapsule, capsule_t)){ 
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }
    

    // get the two matrixs
    cuda_matrix * m = static_cast<cuda_matrix *>(PyCapsule_GetPointer(matrixCapsule, capsule_t));
    
    const size_t n = m->size1;
    // perform logdet_triangular
    switch(m->dtype) {
    case PYCUDA_DOUBLE: //double
        det= cudalib::linalg::logdet_triangular<double>((double *)m->data, n);
        break;
    case PYCUDA_FLOAT: //single
        det = (double) cudalib::linalg::logdet_triangular<float>((float *)m->data, n);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "wrong matrix data type");
        return 0;
    }

    // return 
    return PyFloat_FromDouble(det);
}

// determinant of a matrix, using Cholesky factorization at first
const char * const pyre::extensions::cuda::matrix::determinant__name__ = "matrix_determinant";
const char * const pyre::extensions::cuda::matrix::determinant__doc__ =
    "matrix determinant of a triangular shape";

PyObject *
pyre::extensions::cuda::matrix::
determinant(PyObject *, PyObject * args) {
    // the arguments
    PyObject * matrixCapsule;
    double det;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!:matrix_determinant",
                                  &PyCapsule_Type, &matrixCapsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(matrixCapsule, capsule_t)){ 
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }
    

    // get the two matrixs
    cuda_matrix * m = static_cast<cuda_matrix *>(PyCapsule_GetPointer(matrixCapsule, capsule_t));
    
    const size_t n = m->size1;
    // perform determinant
    switch(m->dtype) {
    case PYCUDA_DOUBLE: //double
        det= (double)cudalib::linalg::determinant<double>((double *)m->data, n);
        break;
    case PYCUDA_FLOAT: //single
        det = cudalib::linalg::determinant<float>((float *)m->data, n);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "wrong matrix data type");
        return 0;
    }

    // return 
    return PyFloat_FromDouble(det);
}
// end of file
