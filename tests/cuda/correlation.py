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
    Test random numbers
    """
    samples = 2**12
    parameters = 4096

    maxerr = 1e-12

    device = cuda.device(0)

    # vector correlation by gpu
    gv1 = cuda.curand.gaussian(loc=1, scale=2, size=samples)
    gv2 = cuda.curand.gaussian(size=samples)
    gcorr = cuda.stats.correlation(gv1, gv2)

    # vector correlation by cpu
    v1 = gv1.copy_to_host()
    v2 = gv2.copy_to_host()
    corr = gsl.stats.correlation(v1, v2)

    # compare the results
    diff = gcorr-corr
    print("vector correlation, cpu/gpu difference", diff)
    assert(abs(diff) < maxerr)


    # create two random matrices
    gm1 = cuda.curand.gaussian(size=(samples, parameters))
    gm2 = cuda.curand.gaussian(size=(samples, parameters))
    # copy to cpu
    m1 = gm1.copy_to_host()
    m2 = gm2.copy_to_host()

    # correlation along row
    gcorr = cuda.stats.correlation(gm1, gm2, axis=0)

    corr = gsl.vector(shape=m1.shape[1])
    for col in range(m1.shape[1]):
        v1 = m1.getColumn(col)
        v2 = m2.getColumn(col)
        corr[col] = gsl.stats.correlation(v1, v2)

    # check difference
    diff = cuda.stats.max_diff(gcorr, cuda.vector(source=corr, dtype=gcorr.dtype))
    print("matrix correlation along row, cpu/gpu max difference", diff)
    assert( diff < maxerr)

    # correlation along column
    gcorr = cuda.stats.correlation(gm1, gm2, axis=1)

    corr = gsl.vector(shape=m1.shape[0])
    for row in range(m1.shape[0]):
        v1 = m1.getRow(row)
        v2 = m2.getRow(row)
        corr[row] = gsl.stats.correlation(v1, v2)

    # check difference
    diff = cuda.stats.max_diff(gcorr, cuda.vector(source=corr, dtype=gcorr.dtype))
    print("matrix correlation along column, cpu/gpu max difference", diff)
    assert( diff < maxerr)

    return

test()

