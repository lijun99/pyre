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
#include "stream.h"

// access to cudalib definitions
#include <pyre/cuda.h>

// allocate a cuda stream
const char * const pyre::extensions::cuda::stream::alloc__name__ = "stream_alloc";
const char * const pyre::extensions::cuda::stream::alloc__doc__ = "allocate a stream";
PyObject *
pyre::extensions::cuda::stream::
alloc(PyObject *, PyObject *args)
{
    // create a stream generator 
    cudaStream_t stream=NULL;
    cudaSafeCall(cudaStreamCreate(&stream));

    // return as a capsule
    return PyCapsule_New(stream, capsule_t, free);
}

// default destructor
void
pyre::extensions::cuda::stream::
free(PyObject * capsule)
{
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) return;
    // get the generator
    cudaStream_t stream =
        static_cast<cudaStream_t>(PyCapsule_GetPointer(capsule, capsule_t));
    
    // deallocate
    cudaSafeCall(cudaStreamDestroy(stream));
    // and return
    return;
}

//end of file
