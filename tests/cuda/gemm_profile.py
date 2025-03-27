# -*- coding: utf-8 -*-
#
# Lijun Zhu (ljzhu@gps.caltech.edu)
#
# (c) 2018-2019 all rights reserved
#

import cuda


def benchmark(m, n, k, device=0, precision='float32', iteration=10):
    """
    benchmarking gemm
    generate C(m,n) = A(m,k) x B(k, n)
    :param device: cuda device to be used
    :param precision: gpu precision 'float32' or 'float64'
    :param iteration: times to repeat gemm for averaging
    """
    # set up device
    device = cuda.device(0)
    cublas_hanlde = device.get_cublas_handle()
    # create a gpu timer (with cudaEvent)
    gtimer = cuda.timer()

    # set up matrices
    matrix_A = cuda.curand.gaussian(size=(m,k), dtype=precision)
    matrix_B = cuda.curand.gaussian(size=(k,n), dtype=precision)
    matrix_C = cuda.matrix(shape=(m,n), dtype=precision)
    # record the start time
    gtimer.start()
    # iterate gemm process
    for i in range(iteration):
        cuda.cublas.gemm(matrix_A, matrix_B, out=matrix_C, handle=cublas_hanlde)

    #record the stop time
    elapsedtime = gtimer.stop()

    # get the average computation time (in s)
    time = elapsedtime/iteration/1e3

    # all done
    return time

def mat_mul(A, B, C, handle, iteration):
    for i in range(iteration):
        cuda.cublas.gemm(A, B, out=C, handle=handle)
    return

def mat_mul_tensor_core(A, B, C, handle, iteration):
    for i in range(iteration):
        cuda.cublas.gemmex(A, B, out=C, handle=handle)
    return

def benchmark2(m, n, k, device=0, precision='float32', iteration=10, use_tensor=False):
    """
    benchmarking gemm
    generate C(m,n) = A(m,k) x B(k, n)
    :param device: cuda device to be used
    :param precision: gpu precision 'float32' or 'float64'
    :param iteration: times to repeat gemm for averaging
    """
    # set up device
    device = cuda.device(0)
    cublas_hanlde = device.get_cublas_handle()
    # create a gpu timer (with cudaEvent)
    gtimer = cuda.timer()

    # set up matrices
    matrix_A = cuda.curand.gaussian(size=(m,k), dtype=precision)
    matrix_B = cuda.curand.gaussian(size=(k,n), dtype=precision)
    matrix_C = cuda.matrix(shape=(m,n), dtype=precision)
    # record the start time

    #record the stop time
    with gtimer.profile():
        if use_tensor:
            mat_mul_tensor_core(matrix_A, matrix_B, matrix_C, cublas_hanlde, iteration)
        else:
            mat_mul(matrix_A, matrix_B, matrix_C, cublas_hanlde, iteration)
    # get the average computation time (in s)
    time = gtimer.elapsed_time/iteration/1e3

    # all done
    return time

def test():

    # check FP64 capability
    n = 6
    flops_ref = (2 ** 14) * (2 ** 14) * (2 ** n) * 2 / (10 ** 12)
    time3 = benchmark2(2 ** n, 2 ** 14, 2 ** 14, precision='float64')
    skip_dp = (flops_ref/time3 < 2.0)

    print("Profiling gemm for two matrices (2**n, 2**14)x(2**14, 2**14)") 
    print("n Tflops(FP16) Tflops(SP) Tflops(DP) ")
    for n in range(1,15):
        flops_ref = (2**14)*(2**14)*(2**n)*2/(10**12)
        time1 = benchmark2(2 ** n, 2 ** 14, 2 ** 14, precision='float32', use_tensor=True)
        time2=benchmark2(2**n, 2**14, 2**14, precision='float32')
        if n <= 6 or not skip_dp :
            time3=benchmark2(2**n, 2**14, 2**14, precision='float64')
            print(f"{n}, {flops_ref/time1:.4f}, {flops_ref/time2:.4f}, {flops_ref/time3:.4f}")
        else:
            print(f"{n}, {flops_ref / time1:.4f}, {flops_ref / time2:.4f}, skip")
    return

test()






