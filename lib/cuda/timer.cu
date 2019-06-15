// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

// my declaration
#include "timer.h"

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

