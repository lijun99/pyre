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

#include "error.h"
#include <cuda_runtime.h>

// place everything in the local namespace
namespace cudalib {
        class cuTimer;
} // of namespace cudalib

// declaration
class cudalib::cuTimer {
public:
    cudaEvent_t _start;
    cudaEvent_t _end;

public:
    // constructor
    /* flags
    #define cudaEventDefault 0x00
    #define cudaEventBlockingSync 0x01
    #define cudaEventDisableTiming 0x02
    #define cudaEventInterprocess 0x04
    */
    cuTimer(unsigned int flags=1)  {
        cudaSafeCall(cudaEventCreateWithFlags(&_start, flags));
        cudaSafeCall(cudaEventCreateWithFlags(&_end, flags));
    }
    // destructor
    ~cuTimer() {
        cudaSafeCall(cudaEventDestroy(_start));
        cudaSafeCall(cudaEventDestroy(_end));
    }
    cuTimer & start();
    cuTimer & stop();
    float duration();
};

#endif
//end of file
