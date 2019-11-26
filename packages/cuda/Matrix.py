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

# the class declaration
class Matrix:
    """
    cuda matrix ( a python wrapper for c/c++ cuda_matrix )

    typedef struct {
        size_t size1; // shape[0]
        size_t size2; // shape[1]
        size_t size;  // total size
        char *data; // pointer to gpu memory
        size_t nbytes; // total bytes
        int dtype; // use numpy type_num
    } cuda_matrix;

    properties:
        shape[2]: shape (size1, size2)
        data: PyCapsule for c/c++ cuda_matrix object
        dtype: data type as in numpy
        size: shape[0]*shape[1]
    """

    from .Vector import Vector as vector

    # traits


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

    def copy_to_device(self, out=None, dtype=None):
        """
        Copy a matrix to another gpu matrix with type conversion support
        :param out: pre-allocated output matrix
        :param dtype: output matrix data type if out is none
        :return: out
        """
        # create an output matrix if not pre-allocated
        if out is None:
            dtype = dtype or self.dtype
            out = Matrix(shape=self.shape, dtype=dtype)
        if self.dtype == out.dtype :
             # same type, use copy
             libcuda.matrix_copy(out.data, self.data)
        else :
            # use copytile interface which supports type conversion
            libcuda.matrix_copytile(out.data, (0,0), self.data, (0,0), out.shape)
        # all done
        return out

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

    # make an alias of view
    submatrix = view

    def tovector(self, start=(0, 0), size=None, out=None):
        """
        view a continuous part or whole matrix as a vector (without data copying)
        :param start: tuple (row, col) as starting element
        :param size: number of elements in vector
        :return: a cuda vector of size size
        """
        # get flattened index
        row, col = start
        start_index = row * self.shape[1] + col
        # get the size
        size = self.size-start_index if size is None else size
        # create a vector if not pre-allocated
        out = self.vector(shape=size, dtype=self.dtype)
        # copy the data
        libcuda.matrix_tovector(out.data, self.data, start_index, size)
        # all done
        return out

    def get_row(self, row=0, out=None):
        """
        get one row
        :param row: row index
        :return: a cuda vector of size=columns
        """
        return self.tovector(start=(row, 0), size=self.shape[1], out=out)

    def set_row(self, src, row=0):
        """
        set one row from a vector
        :param src: cuda vector
        :param row: row index
        :return: self
        """
        # size to be copied
        size = (1, src.shape)
        libcuda.matrix_duplicate_vector(self.data, src.data, row, size, 1)
        return self


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
        copy one or more columns to another matrix, the columns to be copied are specified by indices
        :param dst:
        :param indices:
        :param batch:
        :return:
        """
        batch = batch if batch is not None else dst.shape[0]
        libcuda.matrix_copycols(dst.data, self.data, (batch, indices.shape), indices.data)
        return dst


    def duplicateVector(self, src, size=None, incx=1):
        """
        Copy a vector to first one or few rows to this matrix
        :param src:  cuda vector
        :param size:
        :param incx:
        :return:
        """
        size = size if size is not None else (self.shape[0], src.shape)
        libcuda.matrix_duplicate_vector(self.data, src.data, 0, size, incx)
        return self



    def copy_triangle(self, fill=1):
        """
        for nxn triangular matrix, copy upper triangle (fill=1) to lower, or vice versa(fill=0)
        :param fill: 0,1 for lower/upper filled matrix
        :return: self
        """
        libcuda.matrix_copy_triangle(self.data, fill)
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
        return self

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

    def inverse_cholesky(self, out=None, uplo=1):
        """
        Matrix inverse (in place if out is not provided) for symmetric matrix only
        only the lower, upper part is used for uplo=0,1
        """
        from . import cublas
        from . import cusolverdn

        if out is None:
            out = self
        else:
            out.copy(self)

        handle = cusolverdn.get_current_handle()
        # change notation to cublas column major
        uplo_cblas = cublas.CUBLAS_FILL_MODE_LOWER if uplo == cublas.FillModeUpper else cublas.CUBLAS_FILL_MODE_UPPER

        libcuda.matrix_inverse_cholesky(handle, out.data, uplo_cblas)
        return out

    def inverse(self, out=None):
        """
        Matrix inverse with LU
        :param out: output matrix if different from input
        :param uplo: inverse matrix store mode
        :return: self or out
        """
        from . import cublas
        from . import cusolverdn

        if out is None:
            out = self
        else:
            out.copy(self)

        # get handle
        handle = cusolverdn.get_current_handle()
        libcuda.matrix_inverse_lu_cusolver(handle, out.data)
        return out

    def Cholesky(self, out=None, uplo=1):
        """
        Cholesky decomposition
        :param out: output matrix
        :param uplo: store mode for output, 0/1 = lower/upper triangle
        :return: self or out
        """
        # uplo = 1 for upper
        from . import cublas
        from . import cusolverdn
        # check output matrix
        if out is None:
            out = self
        else:
            out.copy(self)
        # change notation to cublas column major
        uplo_cblas = cublas.CUBLAS_FILL_MODE_LOWER if uplo == cublas.FillModeUpper else cublas.CUBLAS_FILL_MODE_UPPER
        # get a cusolver handle
        handle = cusolverdn.get_current_handle()
        # call extension
        libcuda.matrix_cholesky(handle, out.data, uplo_cblas)
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


    def log_det(self, triangular=False):
        """
        matrix log determinant for real symmetric matrix
        """
        if triangular:
            det = libcuda.matrix_logdet_triangular(self.data)
        else:
            # use Cholesky decomposition, so only real symmetric matrix
            # make a copy as Cholesky changes the matrix
            copy = self.clone()
            copy.Cholesky()
            det = 2.0*libcuda.matrix_logdet_triangular(copy.data)
        # return
        return det

    # statistics

    def amin(self):
        """
        minimum value
        """
        return libcuda.matrix_amin(self.data, self.size, 1)

    def amax(self):
        """
        maximum value
        """
        return libcuda.matrix_amax(self.data, self.size, 1)

    def mean(self, axis=None, out=None):
        """
        mean values along axis=0(row), 1(column), or all elements (None)
        :param axis: int or None, axis along which the means are computed. None for all elements
        :param out: output vector for axis=0,1  vector size = columns(axis=0), rows(axis=1)
        :return:  mean value(s) as a vector for axis=0 or 1, as a float
        """

        # check axis
        if axis == 0: # along row
            # allocate output vector if not present
            out = self.vector(shape=self.shape[1], dtype=self.dtype) if out is None else out
            # call cudalib functions
            libcuda.matrix_mean(self.data, out.data, axis)
        elif axis == 1: # along column
            out = self.vector(shape=self.shape[0], dtype=self.dtype)
            libcuda.matrix_mean(self.data, out.data, axis)
        else: # over all elements
            out = libcuda.matrix_mean_flattened(self.data)
        # all done
        return out

    def mean_sd(self, axis=0, out=None, ddof=1):
        """
        mean and stand deviations along row or column
        :param axis: int or None, axis along which the means are computed. None for all elements
        :param out: tuple of two vectors (mean, sd), vector size is 1 (axis=None),  columns(axis=0), rows(axis=1)
        :param ddof: delta degrees of freedom
        :return: tuple of two vectors
        """

        # check axis
        if axis !=0 and axis !=1:
            raise IndexError("axis is out of range")

        # allocate output vectors if not present
        if out is None:
            # mean, sd over flattened matrix
            if axis is None:
                return
            # to be done
            # mean, sd along row
            elif axis == 0:
                mean = self.vector(shape=self.shape[1], dtype=self.dtype)
                sd = self.vector(shape=self.shape[1], dtype=self.dtype)
            # mean, sd along column
            elif axis == 1:
                mean = self.vector(shape=self.shape[0], dtype=self.dtype)
                sd = self.vector(shape=self.shape[0], dtype=self.dtype)
        else:
            # use pre-allocated vectors
            mean, sd = out
            # assuming correct dimension, skip error checking

        # call cudalib functions
        libcuda.matrix_mean_std(self.data, mean.data, sd.data, axis, ddof)

        # return (mean, sd)
        return mean, sd

    def free(self):
        """
        force releasing gpu memory
        :return:
        """
        libcuda.matrix_dealloc(self.data)
        return

    # mpi support
    def bcast(self, communicator=None, source=0):
        """
        Broadcast the given {vector} from {source} to all tasks in {communicator}
        """
        # normalize the communicator
        if communicator is None:
            # get the mpi package
            import mpi
            # use the world by default
            communicator = mpi.world
        # scatter the data
        libcuda.matrix_bcast(communicator.capsule, source, self.data)
        # and return it
        return self

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

    #properties
    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    # container support
    # in-place arithmetic
    def __iadd__(self, other):
        """
        In-place addition with the elements of {other}
        """
        # if other is a matrix
        if isinstance(other, Matrix):
            libcuda.matrix_iadd(self.data, other.data)
        # or a scalar
        elif isinstance(other, float) or isinstance(other, int):
            libcuda.matrix_iadd_scalar(self.data, float(other))
        else:
            # otherwise, let the interpreter know
            raise NotImplemented
        return self

    def __isub__(self, other):
        """
        In-place subtraction with the elements of {other}
        """
        # if other is a matrix
        if isinstance(other, Matrix):
            libcuda.matrix_isub(self.data, other.data)
        # or a scalar
        elif isinstance(other, float) or isinstance(other, int):
            libcuda.matrix_isub_scalar(self.data, float(other))
        else:
            # otherwise, let the interpreter know
            raise NotImplemented
        return self

    def __imul__(self, other):
        """
        In-place scale with a factor {other}
        """
        # if other is a matrix
        if isinstance(other, Matrix):
            libcuda.matrix_imul(self.data, other.data)
        # or a scalar
        elif isinstance(other, float) or isinstance(other, int):
            libcuda.matrix_imul_scalar(self.data, float(other))
        else:
            # otherwise, let the interpreter know
            raise NotImplemented
        return self


    # private data
    data = None

# end of file
