// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

// my declaration
#include "timer.h"
#include "error.h"

// constructor
cudalib::cuTimer::cuTimer(unsigned int flags)   {
        cudaSafeCall(cudaEventCreateWithFlags(&_start, flags));
        cudaSafeCall(cudaEventCreateWithFlags(&_end, flags));
}

// destructor
cudalib::cuTimer::~cuTimer() {
        cudaSafeCall(cudaEventDestroy(_start));
        cudaSafeCall(cudaEventDestroy(_end));
}

cudalib::cuTimer &
cudalib::cuTimer::start()
{
    cudaSafeCall(cudaEventRecord(_start));
    return *this;
}

cudalib::cuTimer &
cudalib::cuTimer::stop()
{
    cudaSafeCall(cudaEventRecord(_end));
    return *this;
}

float
cudalib::cuTimer::duration() {
    float elapsed_time;
    cudaSafeCall(cudaEventSynchronize(_end));
    cudaSafeCall(cudaEventElapsedTime(&elapsed_time, _start, _end));
    return elapsed_time;
}

//methods are already defined in timer.h
//end of file

