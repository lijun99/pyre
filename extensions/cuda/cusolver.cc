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
#include "cusolver.h"

// local support
#include "vector.h"
#include "matrix.h"
//#include "exceptions.h"
#include "dtypes.h"

// access to cudalib definitions
#include <pyre/cuda.h>



// allocate generator
const char * const pyre::extensions::cuda::cusolverDn::cusolverDnCreate__name__ = "cusolverDnCreate";
const char * const pyre::extensions::cuda::cusolverDn::cusolverDnCreate__doc__ = "allocate a cusolverDn handle";
PyObject *
pyre::extensions::cuda::cusolverDn::
cusolverDnCreate(PyObject *, PyObject *args)
{
    // create a cusolver generator
    cusolverDnHandle_t handle = NULL;
    cusolverSafeCall(cusolverDnCreate(&handle));

    // return as a capsule
    return PyCapsule_New(handle, capsule_t, cusolverDnDestroy);
}

// default destructor
void
pyre::extensions::cuda::cusolverDn::
cusolverDnDestroy(PyObject * capsule)
{
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) return;
    // get the generator
    cusolverDnHandle_t handle =
        static_cast<cusolverDnHandle_t>(PyCapsule_GetPointer(capsule, capsule_t));

    // deallocate
    cusolverSafeCall(cusolverDnDestroy(handle));
    // and return
    return;
}

// set stream
const char * const pyre::extensions::cuda::cusolverDn::cusolverDnSetStream__name__ = "cusolverDnSetStream";
const char * const pyre::extensions::cuda::cusolverDn::cusolverDnSetStream__doc__ = "set cuda stream for a cusolverDn handle";
PyObject *
pyre::extensions::cuda::cusolverDn::
cusolverDnSetStream(PyObject *, PyObject *args)
{
    PyObject * handleCapsule, *streamCapsule;

    // parse the args
    if (!PyArg_ParseTuple(args, "O!O!:cusolverDnSetStream",
                                &PyCapsule_Type, &handleCapsule,
                                &PyCapsule_Type, &streamCapsule))
    {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for cusolverDnSetStream");
        return 0;
    }
    // check capsule types
    if (!PyCapsule_IsValid(handleCapsule, capsule_t) ||
            !PyCapsule_IsValid(streamCapsule, pyre::extensions::cuda::stream::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid capsules for cusolverDnSetStream");
        return 0;
    }

    // cast capsules
    cusolverDnHandle_t handle = static_cast<cusolverDnHandle_t>(PyCapsule_GetPointer(handleCapsule, capsule_t));
    cudaStream_t stream = static_cast<cudaStream_t>(PyCapsule_GetPointer(streamCapsule,
                pyre::extensions::cuda::stream::capsule_t));

    // set stream
    cusolverSafeCall(cusolverDnSetStream(handle, stream));

    // all done
    // return none
    Py_RETURN_NONE;
}

//end of file
