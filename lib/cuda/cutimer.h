// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

// code guard
#ifndef cudalib_cutimer_h
#define cudalib_cutimer_h

#include "cuerror.h"

// place everything in the local namespace
namespace cudalib {
        class cuTimer;
} // of namespace cudalib

// declaration
class cudalib::cuTimer {
private:
    cudaEvent_t _start, _end;
    float _elapsed_time;

public:
    // constructor
    cuTimer()  {
        cudaSafeCall(cudaEventCreate(&_start));
        cudaSafeCall(cudaEventCreate(&_end));
    }
    // destructor
    ~cuTimer() {
        cudaSafeCall(cudaEventDestroy(_start));
        cudaSafeCall(cudaEventDestroy(_end));
    }
    cuTimer & start() {
        cudaSafeCall(cudaEventRecord(_start));
        return *this;
    }
    cuTimer & stop() {
        cudaSafeCall(cudaEventRecord(_end));
        cudaSafeCall(cudaEventSynchronize(_end));
        return *this;
    }
    float duration() {
        cudaSafeCall(cudaEventElapsedTime(&_elapsed_time, _start, _end));
        return _elapsed_time;
    }
};

#endif
//end of file
