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

        handle = handle or cuBlas.get_current_handle()
        n = batch or min(y.size//incy, x.size//incx)

        libcuda.cublas_axpy(handle, n, alpha, x.data, incx, y.data, incy)
        return y

    def gemm(A, B, handle=None, out=None, alpha=1.0, beta=0.0, rows=None,
            transa=0, transb=0):
        """
        Matrix-matrix multiplication (no complex support yet)
        Args: op(A) with shape(m,k), op(B) with shape (k, n) in row major
              op(A) = A if transa=0, else A^T
              rows - only first rows are calculated (rows <=m)
        Returns: out (C) with shape (m, n)
                C = alpha A B + beta C
        """

        handle = handle or cuBlas.get_current_handle()

        if alpha == 0.0 and beta == 1.0 :
            # nothing to do
            return

        # C(m, n) = op(A) (m, k) x op(B) (k, n) in row major
        # to cublas with col major
        # C^T(n,m) = opB^T (n, k) opA^T (k, m)

        # get the dimension
        n = B.shape[1] if transb == 0 else B.shape[0]
        if transa == 0:
            k = A.shape[1]
            m = rows or A.shape[0]
        else:
            k = A.shape[0]
            m = rows or A.shape[1]

        # create a output matrix if not provided
        C = out or Matrix(shape=(m,n), dtype=A.dtype)


        # convert to cublas notations
        ctransa = transb
        ctransb = transa
        cA = B
        cB = A

        cm = n
        ck = k
        cn = m

        # call cublas_gemm
        libcuda.cublas_gemm(handle, ctransa, ctransb,
                            cm, cn, ck, alpha,
                            cA.data, cA.shape[1],
                            cB.data, cB.shape[1],
                            beta,
                            C.data, C.shape[1])

        # all done
        return C

    def gemv(A, x, handle=None, out=None, trans = 0, alpha=1.0, beta=0.0):
        """
        y(out) = alpha op(A) x + beta y
        :param A: matrix (m, n)
        :param x: vector with size= n/m if trans=0/1 (notrans/transpose)
        :param handle: cublas handle
        :param out: vector y with size = m/n if trans=0/1
        :param trans:
        :param alpha:
        :param beta:
        :return: y
        """
        # get a cublas handle if not provided
        handle = handle if handle is not None else cubBlas.get_current_handle()

        # get the dimension in cublas col major
        cm = A.shape[1]  # n
        cn = A.shape[0]  # m
        if trans == 0:
            ctrans = 1
            #xsize = n = cm
            #ysize = m = cn
            y = out or Vector(shape=cn, dtype=A.dtype)
        elif trans == 1:
            ctrans = 0
            #xsize = cn = m
            #ysize = cm = n
            y = out or Vector(shape=cm, dtype=A.dtype)


        # call cublas wrapper
        libcuda.cublas_gemv(handle, ctrans,
                            cm, cn, #m, n
                            alpha,
                            A.data, A.shape[1],
                            x.data, 1,
                            beta,
                            y.data, 1
                            )
        # all done
        return y

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

        handle = handle or cuBlas.get_current_handle()

        # row-major to cublas column-major conversion
        # C= AB (sideleft) -> C^T = B^T A^T  (mxn) = (mxn)x(nxn)
        # C = BA (sideright) -> C^T = A^T B^T (mxn) = (mxm) x (mxn)

        C = out or Matrix(shape=B.shape, dtype=B.dtype)

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

        handle = handle or cuBlas.get_current_handle()

        # row-major to cublas column-major conversion
        # C= AB (sideleft) -> C^T = B^T A^T  (mxn) = (mxn)x(nxn) (sideright)
        # C = BA (sideright) -> C^T = A^T B^T (mxn) = (mxm) x (mxn)(sideleft)

        C = out if out is not None else Matrix(shape=B.shape, dtype=B.dtype)

        # change notations to column major
        side_cblas = cuBlas.CUBLAS_SIDE_RIGHT if side == cuBlas.SideLeft else cuBlas.CUBLAS_SIDE_LEFT
        uplo_cblas = cuBlas.CUBLAS_FILL_MODE_UPPER if uplo == cuBlas.FillModeLower else cuBlas.CUBLAS_FILL_MODE_LOWER

        # determine dimensions
        m_cblas = B.shape[1] #n
        n_cblas = B.shape[0] #m
        lda = A.shape[1]
        ldb = B.shape[1]
        ldc = C.shape[1]

        # call cublas
        libcuda.cublas_symm(handle, side_cblas, uplo_cblas,
            m_cblas, n_cblas, alpha,
            A.data, lda,
            B.data, ldb,
            beta,
            C.data, ldc)
        # return
        return C


# end of file
