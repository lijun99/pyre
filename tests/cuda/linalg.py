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

    precision = 'float32' # or 'float64'
    #### GSL ####

    # make a sigma and init its values
    sigma = gsl.matrix(shape=(parameters, parameters))
    for i in range(parameters):
        for j in range(parameters):
            sigma[i,j] = 1 if i==j else (i+1)*(j+1)*0.001

    

    # cholesky factorization
    sigma_chol = sigma.clone()
    sigma_chol = gsl.linalg.cholesky_decomposition(sigma_chol)

    # create random gaussian samples 
    rng = gsl.rng()
    gaussian = gsl.pdf.gaussian(0, 1, rng)
    random = gsl.matrix(shape=(samples, parameters))
    gaussian.matrix(random)

    # trmm
    jump = random.clone()
    jump = gsl.blas.dtrmm(
            sigma_chol.sideRight, sigma_chol.upperTriangular, sigma_chol.opNoTrans, sigma_chol.nonUnitDiagonal,
            1, sigma_chol, jump)

    #gsl_blas_TYPEtrmm(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, 1, chol, delta);

    # gemm
    product = gsl.matrix(shape =(samples, parameters)) 
    gsl.blas.dgemm(0, 0, 1.0, random, sigma, 0, product) 
    
    # inverse
    sigma_inv = sigma.clone()
    lu = gsl.linalg.LU_decomposition(sigma_inv)
    sigma_inv = gsl.linalg.LU_invert(*lu)
    

    ##### CUDA ######
    device = cuda.manager.device(0)
    
    # copy sigma from cpu to hgpu
    dsigma = cuda.matrix(source=sigma, dtype=precision)

    # Cholesky    
    dsigma_chol = dsigma.clone()
    dsigma_chol = dsigma_chol.Cholesky(uplo=cuda.cublas.FillModeUpper)

    # trmm 
    djump = cuda.matrix(source=random, dtype=precision)
    drandom = cuda.matrix(source=random, dtype=precision)
    djump = cuda.cublas.trmm(dsigma_chol, djump, out=djump, uplo=cuda.cublas.FillModeUpper, side=cuda.cublas.SideRight)

    # gemm
    dproduct = cuda.cublas.gemm(drandom, dsigma)
    
    # inverse
    dsigma_inv = dsigma.clone()
    dsigma_inv.inverse() 


    ### compare ####
    print("compare cpu and gpu results")
    print("input sigma")
    sigma.print()
    dsigma.print()
    
    print("cholesky (GPU only stores upper)")
    sigma_chol.print()
    dsigma_chol.print()

    #random
    print("random")
    random.print()
    drandom.print()

    print("trmm")
    jump.print()
    djump.print()

    print("gemm")
    product.print()
    dproduct.print()

    print("inverse")
    sigma_inv.print()
    dsigma_inv.print()

    print("determinant")
    lu = gsl.linalg.LU_decomposition(sigma_inv)
    det =  gsl.linalg.LU_det(*lu)
    print(det)
    print(dsigma_inv.determinant())

    return
    
test()

# end of file
