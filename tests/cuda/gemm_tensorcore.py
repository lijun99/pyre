# -*- coding: utf-8 -*-
#
# Lijun Zhu (ljzhu@gps.caltech.edu)
#
# (c) 2018-2025 all rights reserved
#

import numpy
from scipy.linalg import blas
import cuda

def test():
    """
    Test gemm by tensor core
    """

    precision = 'float32'
    m = 2**12
    k = 2**14
    n = 2**10
    iterations = 10
    #### GSL ####

    # generate random matrices
    A = numpy.random.rand(m, k).astype(precision)
    B = numpy.random.rand(k, n).astype(precision)

    # cpu gemm
    C = A @ B

    # copy to gpu
    dA = cuda.matrix(source=A, dtype=precision)
    dB = cuda.matrix(source=B, dtype=precision)

    # simt gemm
    dC = cuda.cublas.gemm(dA, dB)

    # tensor core gemm
    dC_tc = cuda.cublas.gemmex(dA, dB)

    ### compare ####
    print("gemm, max relative difference between cpu/gpu results: ",
         cuda.stats.max_relative_error(dC, cuda.matrix(source=C, dtype=precision)))

    print("gemm, max relative difference between gpu FP32/FP16 results: ",
         cuda.stats.max_relative_error(dC, dC_tc))


    return

test()

# end of file
