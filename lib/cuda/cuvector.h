// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2018-2019  all rights reserved
//

// code guard
#ifndef cudalib_cuvector_h
#define cudalib_cuvector_h

// cuda memory capsule
typedef struct {
    size_t size; // length
    char *data; // pointer to gpu memory
    size_t nbytes; // total bytes
    int dtype; // use numpy type_num
} cuda_vector;

#endif

// end of file
