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
    samples = 4096*16
    parameters = 2

    dtype='float32'

    if dtype == 'float32':
        maxerr = 1e-6
    else:
        maxerr = 1e-12

    device=cuda.device(0)

    # vector
    gvector = cuda.curand.gaussian(size=parameters, dtype=dtype)
    gmean = gvector.mean()
    gstd = gvector.std(mean=gmean)

    vector = gvector.copy_to_host(type='gsl')
    mean = vector.mean()
    std = vector.sdev(mean=mean)

    print("vector mean/std difference between cpu/gpu", gmean-mean, gstd-std)
    assert(abs(gmean-mean) < maxerr)

    # matrix
    gmatrix = cuda.curand.gaussian(loc=2.5, scale=2.0, size=(samples, parameters), dtype=dtype)

    # cuda results
    gmean, gsd = gmatrix.mean_sd(axis=0)

    # gsl results
    gslmatrix = gmatrix.copy_to_host(type='gsl')
    gslmean, gslsd = gslmatrix.mean_sd(axis=0)

    print("matrix mean/std max difference between cpu/gpu",
          cuda.stats.max_diff(gmean, cuda.vector(source=gslmean, dtype=dtype)),
          cuda.stats.max_diff(gsd, cuda.vector(source=gslsd, dtype=dtype)))


    print("max value difference between gpu/cpu:", gmatrix.amax()-gslmatrix.max())
    print("min value difference between gpu/cpu:", gmatrix.amin()-gslmatrix.min())

    return

test()

