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
#include "stats.h"
// local support
#include "capsules.h"
#include "exceptions.h"
#include "dtypes.h"
// access to cudalib definitions
#include <pyre/cuda.h>

// types
namespace pyre { namespace extensions { namespace cuda { namespace stats {
    const char * const matrix_capsule_t = pyre::extensions::cuda::matrix::capsule_t;
    const char * const vector_capsule_t = pyre::extensions::cuda::vector::capsule_t;
    using matrix_t = ::cuda_matrix;
    using vector_t = ::cuda_vector;
}}}}


// vector minimum
const char * const pyre::extensions::cuda::stats::vector_amin__name__ = "vector_amin";
const char * const pyre::extensions::cuda::stats::vector_amin__doc__ =
    "minimum value of a vector";

PyObject *
pyre::extensions::cuda::stats::
vector_amin(PyObject *, PyObject * args) {
    // the arguments
    PyObject * vCapsule;
    size_t n, stride;
    double result;

    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!kk:vector_amin",
                                  &PyCapsule_Type, &vCapsule,
                                  &n, &stride);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(vCapsule, vector_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    const vector_t * v = static_cast<const vector_t *>(PyCapsule_GetPointer(vCapsule, vector_capsule_t));

    // for different data type
    switch(v->dtype) {
    case PYCUDA_DOUBLE: //double
        result = cudalib::statistics::min<double>((const double *)v->data, n, stride);
        break;
    case PYCUDA_FLOAT: // float
        result = (double) cudalib::statistics::min<float>((const float *)v->data, n, stride);
            break;
    default:
        PyErr_SetString(PyExc_TypeError, "unsupported vector data type");
        return 0;
    }
    // all done, return
    return PyFloat_FromDouble(result);
}

// vector maximum
const char * const pyre::extensions::cuda::stats::vector_amax__name__ = "vector_amax";
const char * const pyre::extensions::cuda::stats::vector_amax__doc__ =
    "maximum value of a vector";

PyObject *
pyre::extensions::cuda::stats::
vector_amax(PyObject *, PyObject * args) {
    // the arguments
    PyObject * vCapsule;
    size_t n, stride;

    double result;

    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!kk:vector_amax",
                                  &PyCapsule_Type, &vCapsule,
                                  &n, &stride);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(vCapsule, vector_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    const vector_t * v = static_cast<const vector_t *>(PyCapsule_GetPointer(vCapsule, vector_capsule_t));

    // for different data type
    switch(v->dtype) {
    case PYCUDA_DOUBLE: //double
        result = cudalib::statistics::max<double>((const double *)v->data, n, stride);
        break;
    case PYCUDA_FLOAT: // float
        result = (double) cudalib::statistics::max<float>((const float *)v->data, n, stride);
            break;
    default:
        PyErr_SetString(PyExc_TypeError, "unsupported vector data type");
        return 0;
    }
    // all done, return
    return PyFloat_FromDouble(result);
}

// vector sum
const char * const pyre::extensions::cuda::stats::vector_sum__name__ = "vector_sum";
const char * const pyre::extensions::cuda::stats::vector_sum__doc__ =
    "maximum value of a vector";

PyObject *
pyre::extensions::cuda::stats::
vector_sum(PyObject *, PyObject * args) {
    // the arguments
    PyObject * vCapsule;
    size_t n, stride;

    double result;
    long long iresult;

    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!kk:vector_sum",
                                  &PyCapsule_Type, &vCapsule,
                                  &n, &stride);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(vCapsule, vector_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    const vector_t * v = static_cast<const vector_t *>(PyCapsule_GetPointer(vCapsule, vector_capsule_t));

    // for different data type
    switch(v->dtype) {
    case PYCUDA_DOUBLE: //double
        result = cudalib::statistics::sum<double>((const double *)v->data, n, stride);
        return PyFloat_FromDouble(result);
        break;
    case PYCUDA_FLOAT: // float
        result = (double) cudalib::statistics::sum<float>((const float *)v->data, n, stride);
        return PyFloat_FromDouble(result);
        break;
    case PYCUDA_INT:
        iresult = (long long) cudalib::statistics::sum<int>((const int *)v->data, n, stride);
        return PyLong_FromLongLong(iresult);
        break;
    case PYCUDA_ULONGLONG:
        iresult = (long long) cudalib::statistics::sum<unsigned long long>
            ((const unsigned long long *)v->data, n, stride);
        return PyLong_FromLongLong(iresult);
    default:
        PyErr_SetString(PyExc_TypeError, "unsupported vector data type");
        return 0;
    }
    // all done,
}

// vector mean
const char * const pyre::extensions::cuda::stats::vector_mean__name__ = "vector_mean";
const char * const pyre::extensions::cuda::stats::vector_mean__doc__ =
    "mean value of a vector";

PyObject *
pyre::extensions::cuda::stats::
vector_mean(PyObject *, PyObject * args) {
    // the arguments
    PyObject * vCapsule;
    size_t n, stride;

    double result;

    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!kk:vector_mean",
                                  &PyCapsule_Type, &vCapsule,
                                  &n, &stride);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(vCapsule, vector_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the vector
    const vector_t * v = static_cast<const vector_t *>(PyCapsule_GetPointer(vCapsule, vector_capsule_t));

    // for different data type
    switch(v->dtype) {
    case PYCUDA_DOUBLE: //double
        result = cudalib::statistics::mean<double>((const double *)v->data, n, stride);
        break;
    case PYCUDA_FLOAT: // float
        result = (double) cudalib::statistics::mean<float>((const float *)v->data, n, stride);
            break;
    default:
        PyErr_SetString(PyExc_TypeError, "unsupported vector data type");
        return 0;
    }
    // all done, return
    return PyFloat_FromDouble(result);
}

// vector std
const char * const pyre::extensions::cuda::stats::vector_std__name__ = "vector_std";
const char * const pyre::extensions::cuda::stats::vector_std__doc__ =
    "std value of a vector";

PyObject *
pyre::extensions::cuda::stats::
vector_std(PyObject *, PyObject * args) {
    // the arguments
    PyObject * vCapsule;
    double mean;
    size_t n, stride;
    int ddof;

    double result;

    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!dkki:vector_std",
                                  &PyCapsule_Type, &vCapsule,
                                  &mean,
                                  &n, &stride, &ddof);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(vCapsule, vector_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    const vector_t * v = static_cast<const vector_t *>(PyCapsule_GetPointer(vCapsule, vector_capsule_t));

    // for different data type
    switch(v->dtype) {
    case PYCUDA_DOUBLE: //double
        result = cudalib::statistics::std<double>((const double *)v->data, mean, n, stride, ddof);
        break;
    case PYCUDA_FLOAT: // float
        {
        float fmean = (float)mean;
        result = (double) cudalib::statistics::std<float>((const float *)v->data, fmean, n, stride, ddof);
            break;
        }
    default:
        PyErr_SetString(PyExc_TypeError, "unsupported vector data type");
        return 0;
    }
    // all done, return
    return PyFloat_FromDouble(result);
}

// matrix minimum
const char * const pyre::extensions::cuda::stats::matrix_amin__name__ = "matrix_amin";
const char * const pyre::extensions::cuda::stats::matrix_amin__doc__ =
    "minimum value of a matrix";

PyObject *
pyre::extensions::cuda::stats::
matrix_amin(PyObject *, PyObject * args) {
    // the arguments
    PyObject * mCapsule;
    size_t n, stride;

    double result;

    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!kk:matrix_amin",
                                  &PyCapsule_Type, &mCapsule,
                                  &n, &stride);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(mCapsule, matrix_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the matrix
    const matrix_t * m = static_cast<const matrix_t *>(PyCapsule_GetPointer(mCapsule, matrix_capsule_t));

    // for different data type
    switch(m->dtype) {
    case PYCUDA_DOUBLE: //double
        result = cudalib::statistics::min<double>((const double *)m->data, n, stride);
        break;
    case PYCUDA_FLOAT: // float
        result = (double) cudalib::statistics::min<float>((const float *)m->data, n, stride);
            break;
    default:
        PyErr_SetString(PyExc_TypeError, "unsupported matrix/vector data type");
        return 0;
    }
    // all done, return
    return PyFloat_FromDouble(result);
}

// matrix maximum
const char * const pyre::extensions::cuda::stats::matrix_amax__name__ = "matrix_amax";
const char * const pyre::extensions::cuda::stats::matrix_amax__doc__ =
    "maximum value of a matrix";

PyObject *
pyre::extensions::cuda::stats::
matrix_amax(PyObject *, PyObject * args) {
    // the arguments
    PyObject * mCapsule;
    size_t n, stride;

    double result;

    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!kk:matrix_amax",
                                  &PyCapsule_Type, &mCapsule,
                                  &n, &stride);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(mCapsule, matrix_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid matrix/vector capsule");
        return 0;
    }

    // get the matrix
    const matrix_t * m = static_cast<const matrix_t *>(PyCapsule_GetPointer(mCapsule, matrix_capsule_t));

    // for different data type
    switch(m->dtype) {
    case PYCUDA_DOUBLE: //double
        result = cudalib::statistics::max<double>((const double *)m->data, n, stride);
        break;
    case PYCUDA_FLOAT: // float
        result = (double) cudalib::statistics::max<float>((const float *)m->data, n, stride);
            break;
    default:
        PyErr_SetString(PyExc_TypeError, "unsupported matrix data type");
        return 0;
    }
    // all done, return
    return PyFloat_FromDouble(result);
}

// matrix sum (all elements)
const char * const pyre::extensions::cuda::stats::matrix_sum__name__ = "matrix_sum";
const char * const pyre::extensions::cuda::stats::matrix_sum__doc__ =
    "sum of a matrix (all elements)";

PyObject *
pyre::extensions::cuda::stats::
matrix_sum(PyObject *, PyObject * args) {
    // the arguments
    PyObject * mCapsule;
    size_t n, stride;

    double result;

    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!kk:matrix_sum",
                                  &PyCapsule_Type, &mCapsule, &n, &stride);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(mCapsule, matrix_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the matrix
    const matrix_t * m = static_cast<const matrix_t *>(PyCapsule_GetPointer(mCapsule, matrix_capsule_t));

    // for different data type
    switch(m->dtype) {
    case PYCUDA_DOUBLE: //double
        result = cudalib::statistics::sum<double>((const double *)m->data, n, stride);
        break;
    case PYCUDA_FLOAT: // float
        result = (double) cudalib::statistics::sum<float>((const float *)m->data, n, stride);
            break;
    default:
        PyErr_SetString(PyExc_TypeError, "unsupported matrix data type");
        return 0;
    }
    // all done, return
    return PyFloat_FromDouble(result);
}

// mean values of all elements matrix
const char * const pyre::extensions::cuda::stats::matrix_mean_flattened__name__ = "matrix_mean_flattened";
const char * const pyre::extensions::cuda::stats::matrix_mean_flattened__doc__ =
    "mean values a (flattened) matrix ";

PyObject *
pyre::extensions::cuda::stats::
matrix_mean_flattened(PyObject *, PyObject * args) {
    // the arguments
    PyObject * matrixCapsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!:matrix_mean_flattened",
                                  &PyCapsule_Type, &matrixCapsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(matrixCapsule, matrix_capsule_t))
    {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the matrix
    const matrix_t * m = static_cast<const matrix_t *>(PyCapsule_GetPointer(matrixCapsule, matrix_capsule_t));

    // result
    double mean;

    // for different data type
    switch(m->dtype) {
    case PYCUDA_DOUBLE: //double
        mean = cudalib::statistics::mean<double>((const double *)m->data, m->size, 1);
        break;
    case PYCUDA_FLOAT: // float
        mean = (double) cudalib::statistics::mean<double>((const double *)m->data, m->size, 1);
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "unsupported matrix data type");
        return 0;
    }

    // return mean
    return PyFloat_FromDouble(mean);
}

// mean of a batch of vectors stored in matrix
const char * const pyre::extensions::cuda::stats::matrix_mean__name__ = "matrix_mean";
const char * const pyre::extensions::cuda::stats::matrix_mean__doc__ =
    "mean values along row or column of a matrix ";

PyObject *
pyre::extensions::cuda::stats::
matrix_mean(PyObject *, PyObject * args) {
    // the arguments
    PyObject * matrixCapsule, * meanCapsule;
    int axis;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!i:matrix_mean",
                                  &PyCapsule_Type, &matrixCapsule,
                                  &PyCapsule_Type, &meanCapsule,
                                  &axis);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(matrixCapsule, matrix_capsule_t)
        || !PyCapsule_IsValid(meanCapsule, vector_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid matrix/vector capsule");
        return 0;
    }

    // get the matrix
    const matrix_t * m = static_cast<const matrix_t *>(PyCapsule_GetPointer(matrixCapsule, matrix_capsule_t));
    vector_t * mean = static_cast<vector_t *>(PyCapsule_GetPointer(meanCapsule, vector_capsule_t));

    // for different axis
    switch(axis) {
    case 0: // along row
        // for different data type
        switch(m->dtype) {
        case PYCUDA_DOUBLE: //double
            cudalib::statistics::matrix_mean_over_rows<double>(*m, *mean);
            break;
        case PYCUDA_FLOAT: // float
            cudalib::statistics::matrix_mean_over_rows<float>(*m, *mean);
            break;
         default:
             PyErr_SetString(PyExc_TypeError, "unsupported matrix data type");
            return 0;
        }
        break;
    case 1: // along col
        // for different data type
        switch(m->dtype) {
        case PYCUDA_DOUBLE: //double
            cudalib::statistics::matrix_mean_over_cols<double>(*m, *mean);
            break;
        case PYCUDA_FLOAT: // float
            cudalib::statistics::matrix_mean_over_cols<float>(*m, *mean);
            break;
        default:
            PyErr_SetString(PyExc_TypeError, "unsupported matrix data type");
            return 0;
        }
        break;
    default:
        return 0;
    }

    // return none
    Py_RETURN_NONE;
}

// mean, sd of a batch of vectors stored in matrix
const char * const pyre::extensions::cuda::stats::matrix_mean_std__name__ = "matrix_mean_std";
const char * const pyre::extensions::cuda::stats::matrix_mean_std__doc__ =
    "mean values and standard deviations along row or column of a matrix ";

PyObject *
pyre::extensions::cuda::stats::
matrix_mean_std(PyObject *, PyObject * args) {
    // the arguments
    PyObject * matrixCapsule, * meanCapsule, * sdCapsule;
    int axis;
    int ddof;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!O!ii:matrix_mean_std",
                                  &PyCapsule_Type, &matrixCapsule,
                                  &PyCapsule_Type, &meanCapsule,
                                  &PyCapsule_Type, &sdCapsule,
                                  &axis, &ddof);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(matrixCapsule, matrix_capsule_t)
        || !PyCapsule_IsValid(meanCapsule, vector_capsule_t)
        || !PyCapsule_IsValid(sdCapsule, vector_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid matrix/vector capsule");
        return 0;
    }

    // get the matrix
    const matrix_t * m = static_cast<const matrix_t *>(PyCapsule_GetPointer(matrixCapsule, matrix_capsule_t));
    vector_t * mean = static_cast<vector_t *>(PyCapsule_GetPointer(meanCapsule, vector_capsule_t));
    vector_t * sd = static_cast<vector_t *>(PyCapsule_GetPointer(sdCapsule, vector_capsule_t));


    // for different axis
    switch(axis) {
    case 0: // along row
        // for different data type
        switch(m->dtype) {
        case PYCUDA_DOUBLE: //double
            cudalib::statistics::matrix_mean_std_over_rows<double>(*m, *mean, *sd, ddof);
            break;
        case PYCUDA_FLOAT: // float
            cudalib::statistics::matrix_mean_std_over_rows<float>(*m, *mean, *sd, ddof);
            break;
         default:
             PyErr_SetString(PyExc_TypeError, "unsupported matrix data type");
            return 0;
        }
        break;
    case 1: // along col
        // for different data type
        switch(m->dtype) {
        case PYCUDA_DOUBLE: //double
            cudalib::statistics::matrix_mean_std_over_cols<double>(*m, *mean, *sd, ddof);
            break;
        case PYCUDA_FLOAT: // float
            cudalib::statistics::matrix_mean_std_over_cols<float>(*m, *mean, *sd, ddof);
            break;
        default:
            PyErr_SetString(PyExc_TypeError, "unsupported matrix data type");
            return 0;
        }
        break;
    default: // sum over all elements
        // to be done
        return 0;
    }

    // return none
    Py_RETURN_NONE;
}

// vector L1norm
const char * const pyre::extensions::cuda::stats::L1norm__name__ = "L1norm";
const char * const pyre::extensions::cuda::stats::L1norm__doc__ =
    "maximum value of a vector";

PyObject *
pyre::extensions::cuda::stats::
L1norm(PyObject *, PyObject * args) {
    // the arguments
    PyObject * vCapsule;
    size_t n, stride;

    double result;

    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!kk:L1norm",
                                  &PyCapsule_Type, &vCapsule, &n, &stride);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(vCapsule, vector_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    const vector_t * v = static_cast<const vector_t *>(PyCapsule_GetPointer(vCapsule, vector_capsule_t));

    // for different data type
    switch(v->dtype) {
    case PYCUDA_DOUBLE: //double
        result = cudalib::statistics::L1norm<double>((const double *)v->data, n, stride);
        break;
    case PYCUDA_FLOAT: // float
        result = (double) cudalib::statistics::L1norm<float>((const float *)v->data, n, stride);
            break;
    default:
        PyErr_SetString(PyExc_TypeError, "unsupported vector data type");
        return 0;
    }
    // all done, return
    return PyFloat_FromDouble(result);
}

// vector L2norm
const char * const pyre::extensions::cuda::stats::L2norm__name__ = "L2norm";
const char * const pyre::extensions::cuda::stats::L2norm__doc__ =
    "maximum value of a vector";

PyObject *
pyre::extensions::cuda::stats::
L2norm(PyObject *, PyObject * args) {
    // the arguments
    PyObject * vCapsule;
    size_t n, stride;

    double result;

    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!kk:L2norm",
                                  &PyCapsule_Type, &vCapsule, &n, &stride);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(vCapsule, vector_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    const vector_t * v = static_cast<const vector_t *>(PyCapsule_GetPointer(vCapsule, vector_capsule_t));

    // for different data type
    switch(v->dtype) {
    case PYCUDA_DOUBLE: //double
        result = cudalib::statistics::L2norm<double>((const double *)v->data, n, stride);
        break;
    case PYCUDA_FLOAT: // float
        result = (double) cudalib::statistics::L2norm<float>((const float *)v->data, n, stride);
            break;
    default:
        PyErr_SetString(PyExc_TypeError, "unsupported vector data type");
        return 0;
    }
    // all done, return
    return PyFloat_FromDouble(result);
}

// vector Linfnorm
const char * const pyre::extensions::cuda::stats::Linfnorm__name__ = "Linfnorm";
const char * const pyre::extensions::cuda::stats::Linfnorm__doc__ =
    "maximum value of a vector";

PyObject *
pyre::extensions::cuda::stats::
Linfnorm(PyObject *, PyObject * args) {
    // the arguments
    PyObject * vCapsule;
    size_t n, stride;

    double result;

    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!kk:Linfnorm",
                                  &PyCapsule_Type, &vCapsule, &n, &stride);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(vCapsule, vector_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    const vector_t * v = static_cast<const vector_t *>(PyCapsule_GetPointer(vCapsule, vector_capsule_t));

    // for different data type
    switch(v->dtype) {
    case PYCUDA_DOUBLE: //double
        result = cudalib::statistics::Linfnorm<double>((const double *)v->data, n, stride);
        break;
    case PYCUDA_FLOAT: // float
        result = (double) cudalib::statistics::Linfnorm<float>((const float *)v->data, n, stride);
            break;
    default:
        PyErr_SetString(PyExc_TypeError, "unsupported vector data type");
        return 0;
    }
    // all done, return
    return PyFloat_FromDouble(result);
}

// vector vector_covariance
const char * const pyre::extensions::cuda::stats::vector_covariance__name__ = "vector_covariance";
const char * const pyre::extensions::cuda::stats::vector_covariance__doc__ =
    "covariance between two vectors";

PyObject *
pyre::extensions::cuda::stats::
vector_covariance(PyObject *, PyObject * args) {
    // the arguments
    PyObject * v1Capsule, * v2Capsule;
    size_t n, stride;
    int ddof;

    double result;

    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!kki:vector_covariance",
                                  &PyCapsule_Type, &v1Capsule,
                                  &PyCapsule_Type, &v2Capsule,
                                  &n, &stride, &ddof);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if ( !PyCapsule_IsValid(v1Capsule, vector_capsule_t)
        || !PyCapsule_IsValid(v2Capsule, vector_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    const vector_t * v1 = static_cast<const vector_t *>(PyCapsule_GetPointer(v1Capsule, vector_capsule_t));
    const vector_t * v2 = static_cast<const vector_t *>(PyCapsule_GetPointer(v2Capsule, vector_capsule_t));

    // for different data type
    switch(v1->dtype) {
    case PYCUDA_DOUBLE: //double
        result = cudalib::statistics::covariance<double>((const double *)v1->data, (const double *)v2->data, n, stride, ddof);
        break;
    case PYCUDA_FLOAT: // float
        result = (double) cudalib::statistics::covariance<float>((const float *)v1->data, (const float *)v2->data, n, stride, ddof);
            break;
    default:
        PyErr_SetString(PyExc_TypeError, "unsupported vector data type");
        return 0;
    }
    // all done, return
    return PyFloat_FromDouble(result);
}

// vector vector_correlation
const char * const pyre::extensions::cuda::stats::vector_correlation__name__ = "vector_correlation";
const char * const pyre::extensions::cuda::stats::vector_correlation__doc__ =
    "correlation between two vectors";

PyObject *
pyre::extensions::cuda::stats::
vector_correlation(PyObject *, PyObject * args) {
    // the arguments
    PyObject * v1Capsule, * v2Capsule;
    size_t n, stride;

    double result;

    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!kk:vector_correlation",
                                  &PyCapsule_Type, &v1Capsule,
                                  &PyCapsule_Type, &v2Capsule,
                                  &n, &stride);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if ( !PyCapsule_IsValid(v1Capsule, vector_capsule_t)
        || !PyCapsule_IsValid(v2Capsule, vector_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    const vector_t * v1 = static_cast<const vector_t *>(PyCapsule_GetPointer(v1Capsule, vector_capsule_t));
    const vector_t * v2 = static_cast<const vector_t *>(PyCapsule_GetPointer(v2Capsule, vector_capsule_t));

    // for different data type
    switch(v1->dtype) {
    case PYCUDA_DOUBLE: //double
        result = cudalib::statistics::correlation<double>((const double *)v1->data, (const double *)v2->data, n, stride);
        break;
    case PYCUDA_FLOAT: // float
        result = (double) cudalib::statistics::correlation<float>((const float *)v1->data, (const float *)v2->data, n, stride);
            break;
    default:
        PyErr_SetString(PyExc_TypeError, "unsupported vector data type");
        return 0;
    }
    // all done, return
    return PyFloat_FromDouble(result);
}

// covariance between rows or cols of two matrices
const char * const pyre::extensions::cuda::stats::matrix_covariance__name__ = "matrix_covariance";
const char * const pyre::extensions::cuda::stats::matrix_covariance__doc__ =
    "covariance between rows or cols of two matrices";

PyObject *
pyre::extensions::cuda::stats::
matrix_covariance(PyObject *, PyObject * args) {
    // the arguments
    PyObject * m1Capsule, * m2Capsule, * resultCapsule;
    int axis;
    int ddof;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!O!ii:matrix_covariance",
                                  &PyCapsule_Type, &m1Capsule,
                                  &PyCapsule_Type, &m2Capsule,
                                  &PyCapsule_Type, &resultCapsule,
                                  &axis, &ddof);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(m1Capsule, matrix_capsule_t)
        || !PyCapsule_IsValid(m2Capsule, matrix_capsule_t)
        || !PyCapsule_IsValid(resultCapsule, vector_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid matrix/vector capsule");
        return 0;
    }

    // get the matrix
    const matrix_t * m1 = static_cast<const matrix_t *>(PyCapsule_GetPointer(m1Capsule, matrix_capsule_t));
    const matrix_t * m2 = static_cast<const matrix_t *>(PyCapsule_GetPointer(m2Capsule, matrix_capsule_t));
    vector_t * result = static_cast<vector_t *>(PyCapsule_GetPointer(resultCapsule, vector_capsule_t));

    // for different axis
    switch(axis) {
    case 0: // along row
        // for different data type
        switch(m1->dtype) {
        case PYCUDA_DOUBLE: //double
            cudalib::statistics::covariance_batched<double>(
                (double *)result->data, // cov
                (const double *)m1->data, // m1
                (const double *)m2->data, // m2
                m1->size1, // n (number of rows)
                m1->size2, // batch
                1, // stride between batches
                m1->size2, // stride between elements
                ddof // delta degrees of freedom
                );
            break;
        case PYCUDA_FLOAT: // float
            cudalib::statistics::covariance_batched<float>(
                (float *)result->data, // cov
                (const float *)m1->data, // m1
                (const float *)m2->data, // m2
                m1->size1, // n (number of rows)
                m1->size2, // batch
                1, // stride between batches
                m1->size2, // stride between elements
                ddof // delta degrees of freedom
                );
            break;
         default:
             PyErr_SetString(PyExc_TypeError, "unsupported matrix data type");
            return 0;
        }
        break;
    case 1: // along col
        // for different data type
        switch(m1->dtype) {
        case PYCUDA_DOUBLE: //double
            cudalib::statistics::covariance_batched<double>(
                (double *)result->data, // cov
                (const double *)m1->data, // m1
                (const double *)m2->data, // m2
                m1->size2, // n (number of cols)
                m1->size1, // batch
                m1->size2, // stride between batches
                1, // stride between elements
                ddof // delta degrees of freedom
                );
            break;
        case PYCUDA_FLOAT: // float
            cudalib::statistics::covariance_batched<float>(
                (float *)result->data, // cov
                (const float *)m1->data, // m1
                (const float *)m2->data, // m2
                m1->size2, // n (number of cols)
                m1->size1, // batch
                m1->size2, // stride between batches
                1, // stride between elements
                ddof // delta degrees of freedom
                );
            break;
        default:
            PyErr_SetString(PyExc_TypeError, "unsupported matrix data type");
            return 0;
        }
        break;
    default: // sum over all elements
        // to be done
        return 0;
    }

    // return none
    Py_RETURN_NONE;
}

// correlation between rows or cols of two matrices
const char * const pyre::extensions::cuda::stats::matrix_correlation__name__ = "matrix_correlation";
const char * const pyre::extensions::cuda::stats::matrix_correlation__doc__ =
    "correlation between rows or cols of two matrices";

PyObject *
pyre::extensions::cuda::stats::
matrix_correlation(PyObject *, PyObject * args) {
    // the arguments
    PyObject * m1Capsule, * m2Capsule, * resultCapsule;
    int axis;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!O!i:stats_matrix_correlation",
                                  &PyCapsule_Type, &m1Capsule,
                                  &PyCapsule_Type, &m2Capsule,
                                  &PyCapsule_Type, &resultCapsule,
                                  &axis);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(m1Capsule, matrix_capsule_t)
        || !PyCapsule_IsValid(m2Capsule, matrix_capsule_t)
        || !PyCapsule_IsValid(resultCapsule, vector_capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid matrix/vector capsule");
        return 0;
    }

    // get the matrix
    const matrix_t * m1 = static_cast<const matrix_t *>(PyCapsule_GetPointer(m1Capsule, matrix_capsule_t));
    const matrix_t * m2 = static_cast<const matrix_t *>(PyCapsule_GetPointer(m2Capsule, matrix_capsule_t));
    vector_t * result = static_cast<vector_t *>(PyCapsule_GetPointer(resultCapsule, vector_capsule_t));

    // for different axis
    switch(axis) {
    case 0: // along row
        // for different data type
        switch(m1->dtype) {
        case PYCUDA_DOUBLE: //double
            cudalib::statistics::correlation_batched<double>(
                (double *)result->data, // cov
                (const double *)m1->data, // m1
                (const double *)m2->data, // m2
                m1->size1, // n (number of rows)
                m1->size2, // batch
                1, // stride between batches
                m1->size2 // stride between elements
                );
            break;
        case PYCUDA_FLOAT: // float
            cudalib::statistics::correlation_batched<float>(
                (float *)result->data, // cov
                (const float *)m1->data, // m1
                (const float *)m2->data, // m2
                m1->size1, // n (number of rows)
                m1->size2, // batch
                1, // stride between batches
                m1->size2 // stride between elements
                );
            break;
         default:
             PyErr_SetString(PyExc_TypeError, "unsupported matrix data type");
            return 0;
        }
        break;
    case 1: // along col
        // for different data type
        switch(m1->dtype) {
        case PYCUDA_DOUBLE: //double
            cudalib::statistics::correlation_batched<double>(
                (double *)result->data, // cov
                (const double *)m1->data, // m1
                (const double *)m2->data, // m2
                m1->size2, // n (number of cols)
                m1->size1, // batch
                m1->size2, // stride between batches
                1 // stride between elements
                );
            break;
        case PYCUDA_FLOAT: // float
            cudalib::statistics::correlation_batched<float>(
                (float *)result->data, // cov
                (const float *)m1->data, // m1
                (const float *)m2->data, // m2
                m1->size2, // n (number of cols)
                m1->size1, // batch
                m1->size2, // stride between batches
                1 // stride between elements
                );
            break;
        default:
            PyErr_SetString(PyExc_TypeError, "unsupported matrix data type");
            return 0;
        }
        break;
    default: // sum over all elements
        // to be done
        return 0;
    }

    // return none
    Py_RETURN_NONE;
}

// end of file
