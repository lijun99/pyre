// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

// code guard
#ifndef cudalib_cumatrix_h
#define cudalib_cumatrix_h

// cuda matrix
typedef struct {
    size_t size1; // dim[0]
    size_t size2; // dim[1]
    size_t size;  // total size
    char *data; // pointer to gpu memory
    size_t nbytes; // total bytes
    int dtype; // use numpy type_num
} cuda_matrix;

#endif
// end of file
