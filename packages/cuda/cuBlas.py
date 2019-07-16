# -*- coding: utf-8 -*-
#
# Lijun Zhu
# california institute of technology
# (c) 2016-2019  all rights reserved
#

# externals
from . import cuda as libcuda # the extension

from .Matrix import Matrix

class cuBlas:
    """
    Wrapper for cublas lib utitilies
    """

    # definitions from cublas_api.h
    # note CUBLAS uses column-major
    # cublasFillMode_t
    CUBLAS_FILL_MODE_LOWER=0
    CUBLAS_FILL_MODE_UPPER=1
    CUBLAS_FILL_MODE_FULL=2

    # cublasDiagType_t
    CUBLAS_DIAG_NON_UNIT=0
    CUBLAS_DIAG_UNIT=1

    # cublasSideMode_t
    CUBLAS_SIDE_LEFT =0
    CUBLAS_SIDE_RIGHT=1

    # cublasOperation_t
    CUBLAS_OP_N=0
    CUBLAS_OP_T=1
    CUBLAS_OP_C=2
    CUBLAS_OP_HERMITAN=2
    CUBLAS_OP_CONJG=3

    # definitions for cuda.matrix with row-major (or cblas)
    # they may differ from CUBLAS definitions, see examples below
    FillModeLower = 0
    FillModeUpper = 1

    DiagNonUnit = 0
    DiagUnit = 1

    SideLeft = 0
    SideRight = 1

    OpNoTrans = 0
    OpTrans = 1
    # complex not supported yet

    def create_handle():
        """
        create a cublas handle
        """
        handle = libcuda.cublas_alloc()
        return handle

    def get_current_handle():
        # default device handle
        from . import manager
        if manager.current_device is None:
            manager.device(0)
        handle = manager.current_device.get_cublas_handle()
        return handle

    def axpy(alpha, x, y, handle=None, batch=None, incx=1, incy=1):
        """
        axpy : y = alpha x + y
        """

        handle = handle if handle is not None else cuBlas.get_current_handle()
        n = batch if batch is not None else min(y.size//incy, x.size//incx)

        libcuda.cublas_axpy(handle, n, alpha, x.data, incx, y.data, incy)
        return y

    def gemm(A, B, handle=None, out=None, alpha=1.0, beta=0.0, rows=None,
            transa=0, transb=0):
        """
        Matrix-matrix multiplication
        Args: A with shape(m,k), B with shape (k, n)
              rows - only first rows are calculated (rows <=m)
        Returns: out (C) with shape (m, n)
                C = alpha A B + beta C
        """

        if handle is None:
            handle = cuBlas.get_current_handle()

        if alpha == 0.0 and beta == 1.0 :
            # nothing to do
            return

        # C = A B converts to C^T = B^T A^T in cublas
        mc = B.shape[1] # n
        nc = A.shape[0] if rows is None else rows # m
        kc = A.shape[1] # k

        # create a output matrix if not provided
        C = Matrix(shape=(nc, B.shape[1]), dtype=A.dtype) if out is None else out
        # call cublas_gemm
        libcuda.cublas_gemm(handle, transb, transa,
                            mc, nc, kc, alpha,
                            B.data, B.shape[1],
                            A.data, A.shape[1],
                            beta,
                            C.data, C.shape[1])
        return C

    def gemv(A, x, handle=None, out=None, trans = 0, alpha=1.0, beta=0.0):
        """
        y(out) = alpha op(A) x + beta y
        :param A: matrix (rows, cols)
        :param x: vector with size= cols/rows if trans=0/1 (notrans/transpose)
        :param handle: cublas handle
        :param out:
        :param trans:
        :param alpha:
        :param beta:
        :return:
        """
        # get a cublas handle if not provided
        handle = handle if handle is not None else cubBlas.get_current_handle()

        # allocate y/output if not provided
        if out is None:
            out_size = A.shape[1] if trans ==0 else A.shape[0]
            out = Vector(shape=out_size, dtype=A.dtype)
        # reverse the transpose flag
        trans_cblas = cuBlas.CUBLAS_OP_N if trans==1 else cuBlas.CUBLAS_OP_T
        # call cublas wrapper (not that cublas uses column-major)
        libcuda.cublas_gemv(handle, trans,
                            A.shape[1], A.shape[0], #m, n
                            alpha,
                            A.data, A.shape[1],
                            x.data, 1,
                            beta,
                            out.data, 1
                            )
        # all done
        return out

    def trmv(A, x, handle=None,
            uplo=1, #upper
            transa = 0, #NoTrans
            diag = 0, #DiagNonUnit
            incx = 1,
            n = None
            ):
        """
        triangular matrix-vector multiplication x= op(A) x
        Args: A symmetric nxn, x vector n
        Return: x
        """
        if handle is None:
            handle = cuBlas.get_current_handle()

        # change notations to column major
        uplo_cblas = cuBlas.CUBLAS_FILL_MODE_UPPER if uplo == cuBlas.FillModeLower else cuBlas.CUBLAS_FILL_MODE_LOWER
        transa_cblas = transa # no change
        diag_cblas = diag


        # determine dimensions
        lda = A.shape[1]
        n = n if n is not None else A.shape[1]

        # call cublas
        libcuda.cublas_trmv(handle,
            uplo_cblas, transa_cblas, diag_cblas,
            n,
            A.data, lda,
            x.data, incx)
        # return
        return x

    def trmm(A, B, handle=None, out=None, alpha=1.0,
            uplo = 1, #upper
            side = 0, #left
            transa = 0, # NoTrans
            diag = 0 #DiagNonUnit
            ):
        """
        symmetric matrix-matrix multiplication C= A B (Note in blas B = A B)
        Args: if SideLeft A symmetric mxm, B mxn
              if SideRight, A symmetric nxn B mxn
        Return: out(C)  m x n
        """

        if handle is None:
            handle = cuBlas.get_current_handle()

        # row-major to cublas column-major conversion
        # C= AB (sideleft) -> C^T = B^T A^T  (mxn) = (mxn)x(nxn)
        # C = BA (sideright) -> C^T = A^T B^T (mxn) = (mxm) x (mxn)

        C = out if out is not None else Matrix(shape=B.shape, dtype=B.dtype)

        # change notations to column major
        side_cblas = cuBlas.CUBLAS_SIDE_RIGHT if side == cuBlas.SideLeft else cuBlas.CUBLAS_SIDE_LEFT
        uplo_cblas = cuBlas.CUBLAS_FILL_MODE_UPPER if uplo == cuBlas.FillModeLower else cuBlas.CUBLAS_FILL_MODE_LOWER
        transa_cblas = transa # no change
        diag_cblas = diag

        # determine dimensions
        m = B.shape[1]
        n = B.shape[0]
        lda = A.shape[1]
        ldb = B.shape[1]
        ldc = C.shape[1]

        # call cublas
        libcuda.cublas_trmm(handle, side_cblas, uplo_cblas,
            transa_cblas, diag_cblas,
            m, n, alpha,
            A.data, lda,
            B.data, ldb,
            C.data, ldc)
        # return
        return C

    def symv(A, x, handle=None, uplo=1, n=None, alpha=1.0, beta=0.0, out=None):
        """
        symmetric matrix-vector multiplication y = alpha A x + beta y
        Args: A symmetric nxn, x vector n
        Return: x
        """
        if handle is None:
            handle = cuBlas.get_current_handle()

        # change notations to column major
        uplo_cblas = cuBlas.CUBLAS_FILL_MODE_UPPER if uplo == cuBlas.FillModeLower else cuBlas.CUBLAS_FILL_MODE_LOWER


        # determine dimensions
        lda = A.shape[1]
        n = n if n is not None else A.shape[1]

        # get y
        y = out if out is not None else Vector(shape=n, dtype=A.dtype)

        # call cublas
        libcuda.cublas_symv(handle,
                            uplo_cblas,
                            n,
                            alpha,
                            A.data, lda,
                            x.data, 1,
                            beta,
                            out.data, 1)
        # return
        return out

    def symm(A, B, handle=None, out=None, alpha=1.0, beta=0.0,
            uplo = 1, #upper
            side = 0, #left
            ):
        """
        symmetric matrix-matrix multiplication C= A B (Note in blas B = A B)
        Args: if SideLeft A symmetric mxm, B mxn
              if SideRight, A symmetric nxn B mxn
        Return: out(C)  m x n
        """

        if handle is None:
            handle = cuBlas.get_current_handle()

        # row-major to cublas column-major conversion
        # C= AB (sideleft) -> C^T = B^T A^T  (mxn) = (mxn)x(nxn) (sideright)
        # C = BA (sideright) -> C^T = A^T B^T (mxn) = (mxm) x (mxn)(sideleft)

        C = out if out is not None else Matrix(shape=B.shape, dtype=B.dtype)

        # change notations to column major
        side_cblas = cuBlas.CUBLAS_SIDE_RIGHT if side == cuBlas.SideLeft else cuBlas.CUBLAS_SIDE_LEFT
        uplo_cblas = cuBlas.CUBLAS_FILL_MODE_UPPER if uplo == cuBlas.FillModeLower else cuBlas.CUBLAS_FILL_MODE_LOWER

        # determine dimensions
        m = B.shape[1]
        n = B.shape[0]
        lda = A.shape[1]
        ldb = B.shape[1]
        ldc = C.shape[1]

        # call cublas
        libcuda.cublas_symm(handle, side_cblas, uplo_cblas,
            m, n, alpha,
            A.data, lda,
            B.data, ldb,
            beta,
            C.data, ldc)
        # return
        return C


# end of file