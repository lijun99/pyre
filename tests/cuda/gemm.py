### A test for cublas/gemm

import numpy
import cuda

A=numpy.asarray([[2,3],[1.0,2], [4, 2]])
B=numpy.asarray([[2,3,4,5.0],[5,4,3,-1]])
C=numpy.matmul(A, B)

dA=cuda.matrix(source=A)
dB=cuda.matrix(source=B)
handle = cuda.cublas.create_handle()
dC = cuda.cublas.gemm(dA, dB, handle=handle, rows=2)

print("Compare cpu/gpu results (gpu only computes the first two rows as a test)")
print(C)
dC.print()
