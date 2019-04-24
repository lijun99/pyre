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
#include "timer.h"

// access to cudalib definitions
#include <pyre/cuda.h>


// allocate generator
const char * const pyre::extensions::cuda::timer::alloc__name__ = "timer_alloc";
const char * const pyre::extensions::cuda::timer::alloc__doc__ = "allocate a cuda timer";
PyObject *
pyre::extensions::cuda::timer::
alloc(PyObject *, PyObject *args)
{
    // create a timer generator 
    cudalib::cuTimer * timer = new cudalib::cuTimer();
    
    // return as a capsule
    return PyCapsule_New(timer, capsule_t, free);
}

// default destructor
void
pyre::extensions::cuda::timer::
free(PyObject * capsule)
{
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) return;
    // get the timer
    cudalib::cuTimer * timer =
        static_cast<cudalib::cuTimer *>(PyCapsule_GetPointer(capsule, capsule_t));
    
    // deallocate
    timer->~cuTimer();
    // and return
    return;
}

// start
const char * const pyre::extensions::cuda::timer::start__name__ = "timer_start";
const char * const pyre::extensions::cuda::timer::start__doc__ = "start timer";
PyObject *
pyre::extensions::cuda::timer::
start(PyObject *, PyObject *args)
{
    // allocate storage for the arguments
    PyObject * capsule; // timer capsule
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "O!:timer_start", &PyCapsule_Type, &capsule)) {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for timer_start"); 
        return nullptr;
    }
    // check timer generator capsule
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid timer capsule");
        return 0;
    }
    // get the timer
    cudalib::cuTimer * timer =
        static_cast<cudalib::cuTimer *>(PyCapsule_GetPointer(capsule, capsule_t));
    // call 
    timer->start();

    // return None
    Py_RETURN_NONE;
}

// stop
const char * const pyre::extensions::cuda::timer::stop__name__ = "timer_stop";
const char * const pyre::extensions::cuda::timer::stop__doc__ = "stop timer";
PyObject *
pyre::extensions::cuda::timer::
stop(PyObject *, PyObject *args)
{
    // allocate storage for the arguments
    PyObject * capsule; // timer capsule
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "O!:timer_stop", &PyCapsule_Type, &capsule)) {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for timer_stop"); 
        return nullptr;
    }
    // check timer generator capsule
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid timer capsule");
        return 0;
    }
    // get the timer
    cudalib::cuTimer * timer =
        static_cast<cudalib::cuTimer *>(PyCapsule_GetPointer(capsule, capsule_t));
    // call 
    timer->stop();

    // return None
    Py_RETURN_NONE;
}

// elapsed time
const char * const pyre::extensions::cuda::timer::time__name__ = "timer_time";
const char * const pyre::extensions::cuda::timer::time__doc__ = "time timer";
PyObject *
pyre::extensions::cuda::timer::
time(PyObject *, PyObject *args)
{
    // allocate storage for the arguments
    PyObject * capsule; // timer capsule
    double elapsed_time;
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "O!:timer_time", &PyCapsule_Type, &capsule)) {
        // raise an exception
        PyErr_SetString(PyExc_TypeError, "invalid parameters for timer_time"); 
        return nullptr;
    }
    // check timer generator capsule
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid timer capsule");
        return 0;
    }
    // get the timer
    cudalib::cuTimer * timer =
        static_cast<cudalib::cuTimer *>(PyCapsule_GetPointer(capsule, capsule_t));
    // call 
    elapsed_time = (double)timer->duration();

    // return 
    return PyFloat_FromDouble(elapsed_time);
}

//end of file
