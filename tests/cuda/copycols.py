# -*- coding: utf-8 -*-
#
# Lijun Zhu (ljzhu@gps.caltech.edu)
#
# (c) 2018-2019 all rights reserved
#

import numpy
import cuda
import gsl

def test():
    """
    Test Cholesky/trmm/inverse/gemm
    """
    samples = 10
    parameters = 20
    precision = 'float64'

    m = gsl.matrix(shape=(samples, parameters))
    for sample in range(samples):
        for parameter in range(parameters):
            m[sample, parameter] = sample + parameter


    subset = numpy.asarray(list (range(6,8))+ [12, 15] +list(range(18,20)),  dtype='int64')
    gsubset = cuda.vector(source=subset)

    gm = cuda.matrix(source=m, dtype=precision)
    gm_sub = cuda.matrix(shape=(samples, gsubset.shape))
    gm.copycols(dst=gm_sub, indices=gsubset, batch=samples)

    print("A submatrix for indices", subset)
    gm_sub.print()
    
    
    gm.fill(1)
    gm.insert(src=gm_sub, start=(0,1))
    print("insert submatrix back")
    gm.print()

    return

test()
