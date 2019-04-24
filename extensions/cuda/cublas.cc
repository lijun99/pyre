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
    const char * cublasGetErrMsg(cublasStatus_t err);

} } } }


// Get error message
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
    case CUBLAS_STATUS_ALLOC_FAILED:
        message = "CUBLAS_STATUS_ALLOC_FAILED";
        break;
    case CUBLAS_STATUS_INVALID_VALUE:
        message = "CUBLAS_STATUS_INVALID_VALUE";
        break;
    case CUBLAS_STATUS_ARCH_MISMATCH:
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
    cublasSafeCall(cublasCreate(&handle));

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
        static_cast<cublasHandle_t>(PyCapsule_GetPointer(capsule, capsule_t));
    
    // deallocate
    cublasSafeCall(cublasDestroy(handle));
    // and return
    return;
}


// cublas axpy
const char * const pyre::extensions::cuda::cublas::axpy__name__ = "cublas_axpy";
const char * const pyre::extensions::cuda::cublas::axpy__doc__ = "cublas axpy";
PyObject *
pyre::extensions::cuda::cublas::
axpy(PyObject *, PyObject *args)
{
    /*
     * cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
                           const float           *alpha, //host or device
                           const float           *x, int incx, 
                           float                 *y, int incy)
     */
    // allocate storage for the arguments
    PyObject * handleCapsule; // cublas handle capsule
    PyObject * xCapsule, * yCapsule; // cuda vector or matrix
    double alpha;
    int incx, incy, n;
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "O!idO!iO!i:cublas_axpy",
                                &PyCapsule_Type, &handleCapsule,
                                &n, &alpha, 
                                &PyCapsule_Type, &xCapsule, &incx,
                                &PyCapsule_Type, &yCapsule, &incy))
    {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for cublas_axpy"); 
        return nullptr;
    }
    // check cublas handle capsule
    if (!PyCapsule_IsValid(handleCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid cublas handle");
        return 0;
    }
    // get the handle
    cublasHandle_t handle =
        static_cast<cublasHandle_t>(PyCapsule_GetPointer(handleCapsule, capsule_t));

    // check the data capsule type
    if (PyCapsule_IsValid(yCapsule, pyre::extensions::cuda::vector::capsule_t) &&
            PyCapsule_IsValid(xCapsule, pyre::extensions::cuda::vector::capsule_t))
    {
        // get the vector
        cuda_vector * y = static_cast<cuda_vector *>(PyCapsule_GetPointer(yCapsule, pyre::extensions::cuda::vector::capsule_t));
        cuda_vector * x = static_cast<cuda_vector *>(PyCapsule_GetPointer(xCapsule, pyre::extensions::cuda::vector::capsule_t));

        if(y->dtype == PYCUDA_FLOAT){
            float falpha = (float) alpha;
            cublasSafeCall(cublasSaxpy(handle, n, &falpha, (const float *)x->data, incx, (float *)y->data, incy));
        }
        else
            cublasSafeCall(cublasDaxpy(handle, n, &alpha, (const double *)x->data, incx, (double *)y->data, incy));
    }
    else if (PyCapsule_IsValid(yCapsule, pyre::extensions::cuda::matrix::capsule_t) &&
        PyCapsule_IsValid(xCapsule, pyre::extensions::cuda::matrix::capsule_t))
    {
        // get the matrix
        cuda_matrix * y = static_cast<cuda_matrix *>(PyCapsule_GetPointer(yCapsule, pyre::extensions::cuda::matrix::capsule_t));
        cuda_matrix * x = static_cast<cuda_matrix *>(PyCapsule_GetPointer(xCapsule, pyre::extensions::cuda::matrix::capsule_t));

        if(y->dtype == PYCUDA_FLOAT) {
            float falpha = (float) alpha;
            cublasSafeCall(cublasSaxpy(handle, n, &falpha, (const float *)x->data, incx, (float *)y->data, incy));
        }
        else
            cublasSafeCall(cublasDaxpy(handle, n, &alpha, (const double *)x->data, incx, (double *)y->data, incy));
    }
    else { //report an error
        PyErr_SetString(PyExc_TypeError, "invalid vector/matrix type");
        return 0;
    }
    // return None
    Py_RETURN_NONE;
}


// cublas axpy
const char * const pyre::extensions::cuda::cublas::nrm2__name__ = "cublas_nrm2";
const char * const pyre::extensions::cuda::cublas::nrm2__doc__ = "cublas nrm2";
PyObject *
pyre::extensions::cuda::cublas::
nrm2(PyObject *, PyObject *args)
{
    /*
        cublasStatus_t  cublasSnrm2(cublasHandle_t handle, int n,
                            const float           *x, int incx, float  *result)
     */
    // allocate storage for the arguments
    PyObject * handleCapsule; // cublas handle capsule
    PyObject * xCapsule; // cuda vector or matrix
    int incx, n;
    double result;
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "O!iO!id:cublas_nrm2",
                                &PyCapsule_Type, &handleCapsule,
                                &n,
                                &PyCapsule_Type, &xCapsule, &incx,
                                &result))
    {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for cublas_nrm2"); 
        return nullptr;
    }
    // check cublas handle capsule
    if (!PyCapsule_IsValid(handleCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid cublas handle");
        return 0;
    }
    // get the handle
    cublasHandle_t handle =
        static_cast<cublasHandle_t>(PyCapsule_GetPointer(handleCapsule, capsule_t));

    // check the data capsule type
    if (!PyCapsule_IsValid(xCapsule, pyre::extensions::cuda::vector::capsule_t))
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector type");
        return 0;
    }

    // get the vector
    cuda_vector * x = static_cast<cuda_vector *>(PyCapsule_GetPointer(xCapsule, pyre::extensions::cuda::vector::capsule_t));

    switch(x->dtype){
    case PYCUDA_FLOAT:
        float fresult;
        cublasSafeCall(cublasSnrm2(handle, n, (const float *)x->data, incx, &fresult));
        result = (double)fresult;
        break;
    case PYCUDA_DOUBLE:
        cublasSafeCall(cublasDnrm2(handle, n, (const double *)x->data, incx, &result));
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "only float/double are currently supported");
        return 0;
    }

    // return None
    Py_RETURN_NONE;
}

// cublas trmv the triangular matrix-vector multiplication
// x = op ( A ) x 
// A triangular n x n
// x vector n
// op ( A ) = A if  transa == CUBLAS_OP_N A T if  transa == CUBLAS_OP_T

// note cublas uses column major while python/c uses row-major,
// therefore cuda.matrix should be treated as m=col/size2/shape[1] x n=row/size1/shape[0] for cublas
const char * const pyre::extensions::cuda::cublas::trmv__name__ = "cublas_trmv";
const char * const pyre::extensions::cuda::cublas::trmv__doc__ = "cublas trmv";
PyObject *
pyre::extensions::cuda::cublas::
trmv(PyObject *, PyObject *args)
{
    /*
        cublasStatus_t cublasStrmv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const float           *A, int lda,
                           float           *x, int incx)
     */
    // allocate storage for the arguments
    PyObject * handleCapsule; // cublas handle capsule
    int uplo, trans, diag;
    int n;
    PyObject * ACapsule, * xCapsule; // cuda matrix/vector
    int lda, incx;
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "O!iiiiO!iO!i:cublas_trmv",
                                &PyCapsule_Type, &handleCapsule,
                                &uplo, &trans, &diag,
                                &n,
                                &PyCapsule_Type, &ACapsule, &lda,
                                &PyCapsule_Type, &xCapsule, &incx))
    {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for cublas_trmv"); 
        return nullptr;
    }
    // check cublas handle capsule
    if (!PyCapsule_IsValid(handleCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid cublas handle");
        return 0;
    }
    // get the handle
    cublasHandle_t handle =
        static_cast<cublasHandle_t>(PyCapsule_GetPointer(handleCapsule, capsule_t));

    // check the data capsule type
    if (!PyCapsule_IsValid(ACapsule, pyre::extensions::cuda::matrix::capsule_t) ||
            !PyCapsule_IsValid(xCapsule, pyre::extensions::cuda::vector::capsule_t))
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector/matrix type");
        return 0;
    }

    // get the matrix
    cuda_matrix * A = static_cast<cuda_matrix *>(PyCapsule_GetPointer(ACapsule, pyre::extensions::cuda::matrix::capsule_t));
    cuda_vector * x = static_cast<cuda_vector *>(PyCapsule_GetPointer(xCapsule, pyre::extensions::cuda::vector::capsule_t));

    switch(A->dtype) {
    case PYCUDA_FLOAT:
        cublasSafeCall(cublasStrmv(handle,
            (cublasFillMode_t)uplo,
            (cublasOperation_t)trans, (cublasDiagType_t)diag,
            n,
            (const float *)A->data, lda,
            (float *)x->data, incx));
        break; 
    case PYCUDA_DOUBLE:
        cublasSafeCall(cublasDtrmv(handle,
            (cublasFillMode_t)uplo,
            (cublasOperation_t)trans, (cublasDiagType_t)diag,
            n,
            (const double *)A->data, lda,
            (double *)x->data, incx));
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "only float/double are currently supported");
        return 0;
    }
    
    // return None
    Py_RETURN_NONE;
}

// cublas trmm the triangular matrix-matrix multiplication
// C = α op ( A ) B if  side == CUBLAS_SIDE_LEFT α B op ( A ) if  side == CUBLAS_SIDE_RIGHT
// A triangular
// B, C mxn matrics
// op ( A ) = A if  transa == CUBLAS_OP_N A T if  transa == CUBLAS_OP_T

// note cublas uses column major while python/c uses row-major,
// therefore cuda.matrix should be treated as m=col/size2/shape[1] x n=row/size1/shape[0] for cublas
const char * const pyre::extensions::cuda::cublas::trmm__name__ = "cublas_trmm";
const char * const pyre::extensions::cuda::cublas::trmm__doc__ = "cublas trmm";
PyObject *
pyre::extensions::cuda::cublas::
trmm(PyObject *, PyObject *args)
{
    /*
    cublasStatus_t cublasStrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           float                 *C, int ldc)
     */
    // allocate storage for the arguments
    PyObject * handleCapsule; // cublas handle capsule
    int side, uplo, trans, diag;
    int m, n;
    double alpha;
    PyObject * ACapsule, * BCapsule, * CCapsule; // cuda matrix
    int lda, ldb, ldc;
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "O!iiiiiidO!iO!iO!i:cublas_trmm",
                                &PyCapsule_Type, &handleCapsule,
                                &side, &uplo, &trans, &diag,
                                &m, &n, &alpha, 
                                &PyCapsule_Type, &ACapsule, &lda,
                                &PyCapsule_Type, &BCapsule, &ldb,
                                &PyCapsule_Type, &CCapsule, &ldc))
    {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for cublas_trmm"); 
        return nullptr;
    }
    // check cublas handle capsule
    if (!PyCapsule_IsValid(handleCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid cublas handle");
        return 0;
    }
    // get the handle
    cublasHandle_t handle =
        static_cast<cublasHandle_t>(PyCapsule_GetPointer(handleCapsule, capsule_t));

    // check the data capsule type
    if (!PyCapsule_IsValid(ACapsule, pyre::extensions::cuda::matrix::capsule_t) ||
            !PyCapsule_IsValid(BCapsule, pyre::extensions::cuda::matrix::capsule_t) ||
            !PyCapsule_IsValid(CCapsule, pyre::extensions::cuda::matrix::capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector/matrix type");
        return 0;
    }

    // get the matrix
    cuda_matrix * A = static_cast<cuda_matrix *>(PyCapsule_GetPointer(ACapsule, pyre::extensions::cuda::matrix::capsule_t));
    cuda_matrix * B = static_cast<cuda_matrix *>(PyCapsule_GetPointer(BCapsule, pyre::extensions::cuda::matrix::capsule_t));
    cuda_matrix * C = static_cast<cuda_matrix *>(PyCapsule_GetPointer(CCapsule, pyre::extensions::cuda::matrix::capsule_t));

    switch(A->dtype) {
    case PYCUDA_FLOAT: {
        float falpha = (float)alpha;
        cublasSafeCall(cublasStrmm(handle,
            (cublasSideMode_t)side, (cublasFillMode_t)uplo,
            (cublasOperation_t)trans, (cublasDiagType_t)diag,
            m, n, &falpha,
            (const float *)A->data, lda,
            (const float *)B->data, ldb,
            (float *)C->data, ldc));
        break; }
    case PYCUDA_DOUBLE:
        cublasSafeCall(cublasDtrmm(handle,
            (cublasSideMode_t)side, (cublasFillMode_t)uplo,
            (cublasOperation_t)trans, (cublasDiagType_t)diag,
            m, n, &alpha,
            (const double *)A->data, lda,
            (const double *)B->data, ldb,
            (double *)C->data, ldc));
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "only float/double are currently supported");
        return 0;
    }
    
    // return None
    Py_RETURN_NONE;
}


// cublas gemm matrix-matrix multiplication
// C = α op ( A ) op ( B ) + β C 

// note cublas uses column major while python/c uses row-major,
// therefore cuda.matrix should be treated as m=col/size2/shape[1] x n=row/size1/shape[0] for cublas
const char * const pyre::extensions::cuda::cublas::gemm__name__ = "cublas_gemm";
const char * const pyre::extensions::cuda::cublas::gemm__doc__ = "cublas gemm";
PyObject *
pyre::extensions::cuda::cublas::
gemm(PyObject *, PyObject *args)
{
    /*
    cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
     */
    // allocate storage for the arguments
    PyObject * handleCapsule; // cublas handle capsule
    int transa, transb;
    int m, n, k;
    double alpha, beta;
    PyObject * ACapsule, * BCapsule, * CCapsule; // cuda matrix
    int lda, ldb, ldc;
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "O!iiiiidO!iO!idO!i:cublas_gemm",
                                &PyCapsule_Type, &handleCapsule,
                                &transa, &transb,
                                &m, &n, &k,
                                &alpha, 
                                &PyCapsule_Type, &ACapsule, &lda,
                                &PyCapsule_Type, &BCapsule, &ldb,
                                &beta,
                                &PyCapsule_Type, &CCapsule, &ldc))
    {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for cublas_gemm"); 
        return nullptr;
    }
    // check cublas handle capsule
    if (!PyCapsule_IsValid(handleCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid cublas handle");
        return 0;
    }
    // get the handle
    cublasHandle_t handle =
        static_cast<cublasHandle_t>(PyCapsule_GetPointer(handleCapsule, capsule_t));

    // check the data capsule type
    if (!PyCapsule_IsValid(ACapsule, pyre::extensions::cuda::matrix::capsule_t) ||
            !PyCapsule_IsValid(BCapsule, pyre::extensions::cuda::matrix::capsule_t) ||
            !PyCapsule_IsValid(CCapsule, pyre::extensions::cuda::matrix::capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector/matrix type");
        return 0;
    }

    // get the matrix
    cuda_matrix * A = static_cast<cuda_matrix *>(PyCapsule_GetPointer(ACapsule, pyre::extensions::cuda::matrix::capsule_t));
    cuda_matrix * B = static_cast<cuda_matrix *>(PyCapsule_GetPointer(BCapsule, pyre::extensions::cuda::matrix::capsule_t));
    cuda_matrix * C = static_cast<cuda_matrix *>(PyCapsule_GetPointer(CCapsule, pyre::extensions::cuda::matrix::capsule_t));

    switch(A->dtype) {
    case PYCUDA_FLOAT: {
        float falpha = (float)alpha;
        float fbeta = (float)beta;
        cublasSafeCall(cublasSgemm(handle,
            (cublasOperation_t)transa, (cublasOperation_t)transb,
            m, n, k,
            &falpha,
            (const float *)A->data, lda,
            (const float *)B->data, ldb,
            &fbeta,
            (float *)C->data, ldc));
        break; }
    case PYCUDA_DOUBLE:
        cublasSafeCall(cublasDgemm(handle,
            (cublasOperation_t)transa, (cublasOperation_t)transb,
            m, n, k,
            &alpha,
            (const double *)A->data, lda,
            (const double *)B->data, ldb,
            &beta,
            (double *)C->data, ldc));
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "only float/double are currently supported");
        return 0;
    }
    
    // return None
    Py_RETURN_NONE;
}

// cublas symv symmetric matrix-vector multiplication
// y = α A x + β y

// note cublas uses column major while python/c uses row-major,
// therefore cuda.matrix should be treated as m=col/size2/shape[1] x n=row/size1/shape[0] for cublas
const char * const pyre::extensions::cuda::cublas::symv__name__ = "cublas_symv";
const char * const pyre::extensions::cuda::cublas::symv__doc__ = "cublas symv";
PyObject *
pyre::extensions::cuda::cublas::
symv(PyObject *, PyObject *args)
{
    /*
    cublasStatus_t cublasSsymv(cublasHandle_t handle, cublasFillMode_t uplo,
                           int n, const float           *alpha,
                           const float           *A, int lda,
                           const float           *x, int incx, const float           *beta,
                           float           *y, int incy)
     */
    // allocate storage for the arguments
    PyObject * handleCapsule; // cublas handle capsule
    int uplo;
    int n;
    double alpha, beta;
    PyObject * ACapsule, * xCapsule, * yCapsule; // cuda matrix/vector
    int lda, incx, incy;
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "O!iidO!iO!idO!i:cublas_symv",
                                &PyCapsule_Type, &handleCapsule,
                                &uplo,
                                &n,
                                &alpha, 
                                &PyCapsule_Type, &ACapsule, &lda,
                                &PyCapsule_Type, &xCapsule, &incx,
                                &beta,
                                &PyCapsule_Type, &yCapsule, &incy))
    {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for cublas_symv"); 
        return nullptr;
    }
    // check cublas handle capsule
    if (!PyCapsule_IsValid(handleCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid cublas handle");
        return 0;
    }
    // check the data capsule type
    if (!PyCapsule_IsValid(ACapsule, pyre::extensions::cuda::matrix::capsule_t) ||
            !PyCapsule_IsValid(xCapsule, pyre::extensions::cuda::vector::capsule_t) ||
            !PyCapsule_IsValid(yCapsule, pyre::extensions::cuda::vector::capsule_t) )
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector/matrix type");
        return 0;
    }
    // get the handle
    cublasHandle_t handle =
        static_cast<cublasHandle_t>(PyCapsule_GetPointer(handleCapsule, capsule_t));
        
    // get the matrix
    cuda_matrix * A = static_cast<cuda_matrix *>(PyCapsule_GetPointer(ACapsule, pyre::extensions::cuda::matrix::capsule_t));
    cuda_vector * x = static_cast<cuda_vector *>(PyCapsule_GetPointer(xCapsule, pyre::extensions::cuda::vector::capsule_t));
    cuda_vector * y = static_cast<cuda_vector *>(PyCapsule_GetPointer(yCapsule, pyre::extensions::cuda::vector::capsule_t));

    switch(A->dtype) {
    case PYCUDA_FLOAT: {
        float falpha = (float)alpha;
        float fbeta = (float)beta;
        cublasSafeCall(cublasSsymv(handle, (cublasFillMode_t) uplo, 
            n, &falpha,
            (const float *)A->data, lda,
            (const float *)x->data, incx,
            &fbeta,
            (float *)y->data, incy));
        break; }
    case PYCUDA_DOUBLE:
        cublasSafeCall(cublasDsymv(handle, (cublasFillMode_t) uplo, 
            n, &alpha,
            (const double *)A->data, lda,
            (const double *)x->data, incx,
            &beta,
            (double *)y->data, incy));
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "only float/double are currently supported");
        return 0;
    }
    
    // return None
    Py_RETURN_NONE;
}

// cublas syr symmetric matrix-vector multiplication
// A = alpha x x^T + A

// note cublas uses column major while python/c uses row-major,
// therefore cuda.matrix should be treated as m=col/size2/shape[1] x n=row/size1/shape[0] for cublas
const char * const pyre::extensions::cuda::cublas::syr__name__ = "cublas_syr";
const char * const pyre::extensions::cuda::cublas::syr__doc__ = "cublas syr";
PyObject *
pyre::extensions::cuda::cublas::
syr(PyObject *, PyObject *args)
{
    /*
    cublasStatus_t cublasSsyr(cublasHandle_t handle, cublasFillMode_t uplo,
                          int n, const float           *alpha,
                          const float           *x, int incx, float           *A, int lda)
     */
    // allocate storage for the arguments
    PyObject * handleCapsule; // cublas handle capsule
    int uplo;
    int n;
    double alpha;
    PyObject * xCapsule, * ACapsule; // cuda matrix/vector
    int lda, incx;
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "O!iidO!iO!i:cublas_syr",
                                &PyCapsule_Type, &handleCapsule,
                                &uplo,
                                &n,
                                &alpha, 
                                &PyCapsule_Type, &xCapsule, &incx,
                                &PyCapsule_Type, &ACapsule, &lda))
    {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for cublas_syr"); 
        return nullptr;
    }
    // check cublas handle capsule
    if (!PyCapsule_IsValid(handleCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid cublas handle");
        return 0;
    }
    // check the data capsule type
    if (!PyCapsule_IsValid(ACapsule, pyre::extensions::cuda::matrix::capsule_t) ||
            !PyCapsule_IsValid(xCapsule, pyre::extensions::cuda::vector::capsule_t))
    {
        PyErr_SetString(PyExc_TypeError, "invalid vector/matrix type");
        return 0;
    }
    // get the handle
    cublasHandle_t handle =
        static_cast<cublasHandle_t>(PyCapsule_GetPointer(handleCapsule, capsule_t));
        
    // get the matrix
    cuda_matrix * A = static_cast<cuda_matrix *>(PyCapsule_GetPointer(ACapsule, pyre::extensions::cuda::matrix::capsule_t));
    cuda_vector * x = static_cast<cuda_vector *>(PyCapsule_GetPointer(xCapsule, pyre::extensions::cuda::vector::capsule_t));

    switch(A->dtype) {
    case PYCUDA_FLOAT: {
        float falpha = (float)alpha;
        cublasSafeCall(cublasSsyr(handle, (cublasFillMode_t) uplo, 
            n, &falpha,
            (const float *)x->data, incx,
            (float *)A->data, lda));
        break; }
    case PYCUDA_DOUBLE:
        cublasSafeCall(cublasDsyr(handle, (cublasFillMode_t) uplo, 
            n, &alpha,
            (const double *)x->data, incx,
            (double *)A->data, lda));
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "only float/double are currently supported");
        return 0;
    }
    
    // return None
    Py_RETURN_NONE;
}

//end of file
