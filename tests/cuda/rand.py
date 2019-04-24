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
    samples = 8
    parameters = 4
    
    vgaussian = cuda.curand.gaussian(size=samples)
    mgaussian = cuda.curand.gaussian(size=(samples, parameters))
    
    print("gaussian vector/matrix")
    vgaussian.print()
    mgaussian.print()
    
    vuniform = cuda.vector(shape=samples)    
    vuniform = cuda.curand.uniform(out=vuniform)
    support =(-2, 2)
    srange = support[1]-support[0]
    vuniform*=srange
    vuniform+=cuda.vector(shape=samples).fill(support[0])
    
    muniform = cuda.curand.uniform(size=(samples, parameters), dtype='float32')
    print("uniform vector/matrix")
    vuniform.print()
    muniform.print()
    
    
    return
    
test()
    
