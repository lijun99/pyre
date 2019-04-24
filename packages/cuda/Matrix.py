# -*- coding: utf-8 -*-
#
# Lijun Zhu
# california institute of technology
# (c) 2016-2019  all rights reserved
#


# externals
import numpy
import gsl
from . import cuda as libcuda # the extension
from . import Device

# the class declaration
class Matrix:
    """
    Cuda matrix
    """

    # class methods
    def copy_to_host(self, target=None, type='gsl'):
        """
        copy cuda matrix to host (gsl or numpy)
        gsl.matrix is double precison only
        numpy.ndarray can be any type 
        """
        # if target is not given
        if target is None:
            if type == 'gsl':
                target = gsl.matrix(shape=self.shape)
                libcuda.matrix_togsl(target.data, self.data)
            else: #numpy
                target = numpy.ndarray(shape=self.shape, dtype=self.dtype)
                libcuda.matrix_tonumpy(target, self.data)    
        # if target is a pre-allocated gsl.matrix
        elif isinstance(target, gsl.matrix):
            libcuda.matrix_togsl(target.data, self.data)
        # assume numpy ndarray for rest cases
        else: 
            libcuda.matrix_tonumpy(target, self.data)

        # return    
        return target
        
    def copy_from_host(self, source):
        """
        copy from a gsl(host) matrix  
        """
        # assuming pre-allocated with right shape
        if isinstance(source, gsl.matrix):
            libcuda.matrix_fromgsl(self.data, source.data)
        elif isinstance(source, numpy.ndarray):
            libcuda.matrix_fromnumpy(self.data, source)
        return self

    def copy(self, other):
        """
        copy data from another matrix
        """
        libcuda.matrix_copy(self.data, other.data)
        return self
        
    def clone(self):
        """
        clone to a new matrix
        """
        other=Matrix(shape=self.shape, dtype=self.dtype)
        libcuda.matrix_copy(other.data, self.data)
        return other

    def view(self, out=None, start=(0,0), size=None):
        """
        copy a submatrix with size=(m,n) from start=(ms, ns)
        """
        # create a matrix if output is not provided
        if out is None:
            out = Matrix(shape=size, dtype=self.dtype)
        # call the wrapper
        libcuda.matrix_copytile(out.data, (0,0), self.data, start, out.shape)
        # return submatrix
        return out
        
    # make an alias
    submatrix = view
    
    def insert(self, src, start=(0,0), shape=None):
        """
        insert (copy) a matrix from position start 
        """
        shape = src.shape if shape is None else shape
        libcuda.matrix_copytile(self.data, start, src.data, (0, 0), shape)
        return self

    def copytile(self, src, start=(0,0), src_start=(0,0), shape=None):
        """
        copy a tile of matrix from src        
        """
        shape = (min(self.shape[0]-start[0], src.shape[0]-src_start[0]),
                min(self.shape[1]-start[1], src.shape[1]-src_start[1])) if shape is None else shape
        libcuda.matrix_copytile(self.data, start, src.data, src_start, shape)
        return self

    def copycols(self, dst, indices, batch=None):
        """
        copy selected columns as listed in indices
        """
        batch = batch if batch is not None else dst.shape[0]
        libcuda.matrix_copycols(dst.data, self.data, (batch, indices.shape), indices.data)
        return dst
        
    
    def duplicateVector(self, src, size=None, incx=1):
        """
        duplicate batch copies of vector from src
        """
        size = size if size is not None else (self.shape[0], src.shape)
        libcuda.matrix_duplicate_vector(self.data, src.data, size, incx)
        return self
        
        
    def zero(self):
        """
        initialize all elements to 0
        """
        libcuda.matrix_zero(self.data)
        return self
        
    def fill(self, value):
        """
        set all elements to a given value
        """
        libcuda.matrix_fill(self.data, value)
        return self
        
    def print(self):
        """
        print elements by converting to gsl(host) matrix at first
        """
        print(self.copy_to_host(type='numpy'))
        return

 
    def transpose(self, out=None):
        """
        transpose M(m,n)-> MT(n,m)
        """
        # create a matrix with rows/cols interchanged
        if out is None:
            out = Matrix(shape=(self.shape[1], self.shape[0]), dtype=self.dtype)
        # call the wrapper
        libcuda.matrix_transpose(out.data, self.data)
        # return 
        return out

    def inverse(self, out=None, uplo=1):
        """
        Matrix inverse (in place if out is not provided) for symmetric matrix only
        only the lower, upper part is used for uplo=0,1
        """
        from . import cublas

        if out is None:
            out = self
        else:
            out.copy(self)

        # change notation to cublas column major
        uplo_cblas = cublas.CUBLAS_FILL_MODE_LOWER if uplo == cublas.FillModeUpper else cublas.CUBLAS_FILL_MODE_UPPER
        
        libcuda.matrix_inverse_symm(out.data, uplo_cblas)
        return out

    def Cholesky(self, out=None, uplo=1):
        """
        Cholesky decomposition 
        """
        # uplo = 1 for upper         
        from . import cublas
        # check output matrix
        if out is None:
            out = self
        else:
            out.copy(self)
        # change notation to cublas column major
        uplo_cblas = cublas.CUBLAS_FILL_MODE_LOWER if uplo == cublas.FillModeUpper else cublas.CUBLAS_FILL_MODE_UPPER 
        # call extension
        libcuda.matrix_cholesky(out.data, uplo_cblas)
        # return
        return out


    def determinant(self, triangular=False):
        """
        matrix determinant for real symmetric matrix
        """
        if triangular:
            det = libcuda.matrix_determinant_triangular(self.data)
        else:
            # use Cholesky decomposition, so only real symmetric matrix
            # make a copy as Cholesky changes the matrix
            copy = self.clone()
            det = libcuda.matrix_determinant(copy.data)
        # return
        return det

    # meta methods
    def __init__(self, shape=(1,1), source=None, dtype="float64", **kwds):
        # create a cuda Matrix from gsl(host) Matrix
        if source is not None:
            if isinstance(source, gsl.matrix):
                self.shape=source.shape
                self.size=self.shape[0]*self.shape[1]
                self.dtype = numpy.dtype(dtype)
                self.nbytes = self.dtype.itemsize*self.size
                self.data = libcuda.matrix_alloc(self.shape, self.nbytes, self.dtype.num)
                libcuda.matrix_fromgsl(self.data, source.data)
            elif isinstance(source, numpy.ndarray):
                self.shape=source.shape
                self.size=self.shape[0]*self.shape[1]
                self.dtype = source.dtype
                self.nbytes = self.dtype.itemsize*self.size
                self.data = libcuda.matrix_alloc(self.shape, self.nbytes, self.dtype.num)
                libcuda.matrix_fromnumpy(self.data, source)
            else:
                raise NotImplementedError("only sources from gsl/numpy are supported")
        # create a new one
        else:
            # check shape
            if not isinstance(shape, tuple):
                raise Exception("shape should be a 2-element tuple (m,n), the input shape was: {}.".format(shape))
                return
            if len(shape)!=2:
                raise Exception("shape should be a 2-element tuple (m,n), the input shape was: {}.".format(shape))
                return 
            # store
            self.shape = tuple(map(int, shape))
            self.size=self.shape[0]*self.shape[1]
            self.dtype = numpy.dtype(dtype)
            self.nbytes = self.dtype.itemsize*self.size
            self.data = libcuda.matrix_alloc(self.shape, self.nbytes, self.dtype.num)
        # all done
        return

    # container support

    # in-place arithmetic
    def __iadd__(self, other):
        """
        In-place addition with the elements of {other}
        """
        # if other is a matrix
        if isinstance(other, Matrix):
            libcuda.matrix_iadd(self.data, other.data)
            return self
        # otherwise, let the interpreter know
        raise NotImplemented
        
    def __isub__(self, other):
        """
        In-place subtraction with the elements of {other}
        """
        # if other is a matrix
        if isinstance(other, Matrix):
            libcuda.matrix_isub(self.data, other.data)
            return self
        # otherwise, let the interpreter know
        raise NotImplemented

    def __imul__(self, other):
        """
        In-place scale with a factor {other}
        """
        # if other is a matrix
        if isinstance(other, float) or isinstance(other, int):
            libcuda.matrix_imul(self.data, float(other))
            return self
        # otherwise, let the interpreter know
        raise NotImplemented

    # private data
    data = None


# end of file
