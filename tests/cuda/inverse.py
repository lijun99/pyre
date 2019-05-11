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
    Test matrix inverse
    """
    samples = 10
    parameters = 32

    precision = 'float32' # or 'float64'
    #### GSL ####

    # make a sigma and init its values
    sigma = gsl.matrix(shape=(parameters, parameters))
    for i in range(parameters):
        for j in range(parameters):
            sigma[i,j] = 1 if i==j else (i+1)*(j+1)*0.0001

    # inverse
    sigma_inv = sigma.clone()
    lu = gsl.linalg.LU_decomposition(sigma_inv)
    sigma_inv = gsl.linalg.LU_invert(*lu)


    ##### CUDA ######
    device = cuda.manager.device(0)

    # copy sigma from cpu to hgpu
    dsigma = cuda.matrix(source=sigma, dtype=precision)

    # inverse
    dsigma_inv = dsigma.clone()
    dsigma_inv.inverse_cholesky()


    ### compare ####
    print("inverse")
    sigma_inv.print()
    dsigma_inv.print()

    return

test()

# end of file
