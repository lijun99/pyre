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
#include "mpi.h"
#include "vector.h"
#include "matrix.h"
// local support
#include "exceptions.h"
#include "dtypes.h"
// so I can build reasonable error messages
#include <sstream>

// access to cudalib definitions
#include <pyre/cuda.h>

// the external libraries
#include <mpi.h>
// the pyre mpi library
#include <pyre/mpi.h>
#include <pyre/mpi/capsules.h>

#include <iostream>

// broadcast a vector
const char * const pyre::extensions::cuda::vector::bcast__name__ = "vector_bcast";
const char * const pyre::extensions::cuda::vector::bcast__doc__ = "bcast a cuda vector";

// @note: current implementation assumes normal mpi, i.e., to employ a cpu array as buffer
// for cuda aware MPI, one can bcast gpu data directly (via GPUDirect).
PyObject *
pyre::extensions::cuda::vector::bcast(PyObject *, PyObject * args)
{
    // place holders
    int source;
    PyObject *communicatorCapsule, *vectorCapsule;

    // parse the argument list
    if (!PyArg_ParseTuple(
                          args,
                          "O!iO!:vector_bcast",
                          &PyCapsule_Type, &communicatorCapsule,
                          &source,
                          &PyCapsule_Type, &vectorCapsule
                          )) {
        return 0;
    }
    // check the communicator capsule
    if (!PyCapsule_IsValid(communicatorCapsule, ::mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(communicatorCapsule, ::mpi::communicator::capsule_t));

    // vector
    cuda_vector * vector = static_cast<cuda_vector *>(PyCapsule_GetPointer(vectorCapsule, capsule_t));
    // get its size in bytes
    size_t nbytes = vector->nbytes;

    // allocate a cpu buffer
    char * buffer = (char *)malloc(nbytes);

    // copy data from gpu to cpu
    if (comm->rank() == source) {
        cudaSafeCall(cudaMemcpy(buffer, vector->data, nbytes, cudaMemcpyDefault));
    }

    // broadcast the data
    int status = MPI_Bcast(buffer, nbytes, MPI_BYTE, source, comm->handle());

    // check the return code
    if (status != MPI_SUCCESS) {
        // and throw an exception if anything went wrong
        PyErr_SetString(PyExc_RuntimeError, "MPI_Bcast failed");
        return 0;
    }

    // copy data from cpu buffer to gpu
    if (comm->rank() != source) {
        cudaSafeCall(cudaMemcpy(vector->data, buffer, nbytes, cudaMemcpyDefault));
    }

    //deallocate buffer
    std::free(buffer);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// broadcast a matrix
const char * const pyre::extensions::cuda::matrix::bcast__name__ = "matrix_bcast";
const char * const pyre::extensions::cuda::matrix::bcast__doc__ = "bcast a cuda matrix";

// @note: current implementation assumes normal mpi, i.e., to employ a cpu array as buffer
// for cuda aware MPI, one can bcast gpu data directly (via GPUDirect).
PyObject *
pyre::extensions::cuda::matrix::bcast(PyObject *, PyObject * args)
{
    // place holders
    int source;
    PyObject *communicatorCapsule, *matrixCapsule;

    // parse the argument list
    if (!PyArg_ParseTuple(
                          args,
                          "O!iO!:matrix_bcast",
                          &PyCapsule_Type, &communicatorCapsule,
                          &source,
                          &PyCapsule_Type, &matrixCapsule
                          )) {
        return 0;
    }
    // check the communicator capsule
    if (!PyCapsule_IsValid(communicatorCapsule, ::mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(communicatorCapsule, ::mpi::communicator::capsule_t));

    // matrix
    cuda_matrix * matrix = static_cast<cuda_matrix *>(PyCapsule_GetPointer(matrixCapsule, capsule_t));
    // get its size in bytes
    size_t nbytes = matrix->nbytes;

    // allocate a cpu buffer
    char * buffer = (char *)malloc(nbytes);

    // copy data from gpu to cpu
    if (comm->rank() == source) {
        cudaSafeCall(cudaMemcpy(buffer, matrix->data, nbytes, cudaMemcpyDefault));
    }

    // broadcast the data
    int status = MPI_Bcast(buffer, nbytes, MPI_BYTE, source, comm->handle());

    // check the return code
    if (status != MPI_SUCCESS) {
        // and throw an exception if anything went wrong
        PyErr_SetString(PyExc_RuntimeError, "MPI_Bcast failed");
        return 0;
    }

    // copy data from cpu buffer to gpu
    if (comm->rank() != source) {
        cudaSafeCall(cudaMemcpy(matrix->data, buffer, nbytes, cudaMemcpyDefault));
    }

    //deallocate buffer
    ::free(buffer);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// end of file
