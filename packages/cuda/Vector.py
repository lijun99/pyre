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
class Vector:
    """
    cuda vector (a python wrapper for c/c++ cuda_vector)
    typedef struct {
        size_t size; // length
        char *data; // pointer to gpu memory
        size_t nbytes; // total bytes
        int dtype; // use numpy type_num
    } cuda_vector;
    """

    # if any cuda method is not available, a numpy (cpu) method can be used as follows
    # v = gv.copy_to_host(type='numpy')
    # v.numpy_method()

    # class methods
    def copy_to_host(self, target=None, type='gsl'):
        """
        copy cuda vector to host (gsl or numpy)vector
        gsl.vector is double precison only
        numpy.ndarray can be any type
        """
        # if target is not given
        if target is None:
            if type == 'gsl':
                target = gsl.vector(shape=self.shape)
                libcuda.vector_togsl(target.data, self.data)
            else: #numpy
                target = numpy.ndarray(shape=self.shape, dtype=self.dtype)
                libcuda.vector_tonumpy(target, self.data)
        # if target is a pre-allocated gsl.vector
        elif isinstance(target, gsl.vector):
            libcuda.vector_togsl(target.data, self.data)
        # assume numpy ndarray for rest cases
        else:
            libcuda.vector_tonumpy(target, self.data)

        # return
        return target

    def copy_from_host(self, source):
        """
        copy from a host (gsl or numpy) vector
        """
        # assuming pre-allocated with right shape
        if isinstance(source, gsl.vector):
            libcuda.vector_fromgsl(self.data, source.data)
        elif isinstance(source, numpy.ndarray):
            libcuda.vector_fromnumpy(self.data, source)
        return

    def copy(self, other):
        """
        copy data from another vector
        """
        libcuda.vector_copy(self.data, other.data)
        return self

    def clone(self):
        """
        clone to a new vector
        """
        other=Vector(shape=self.shape, dtype=self.dtype)
        libcuda.vector_copy(other.data, self.data)
        return other

    def zero(self):
        """
        initialize all elements to 0
        """
        libcuda.vector_zero(self.data)
        return self

    def fill(self, value):
        """
        set all elements to a given value
        """
        libcuda.vector_fill(self.data, value)
        return self

    def print(self):
        """
        print elements by converting to numpy ndarray
        """
        host = self.copy_to_host(type='numpy')
        print(host)

        return

    # statistics
    def sum(self):
        """
        summation
        """
        return libcuda.vector_sum(self.data, self.shape, 1)

    def amin(self):
        """
        minimum value
        """
        return libcuda.vector_amin(self.data, self.shape, 1)

    def amax(self):
        """
        maximum value
        """
        return libcuda.vector_amax(self.data, self.shape, 1)

    def mean(self):
        """
        mean value
        """
        # capsule, size, stride
        return libcuda.vector_mean(self.data, self.shape, 1)

    def std(self, mean=None, ddof=1):
        """
        standard deviation
        :param mean: mean value
        :param ddof: delta degrees of freedom, or the dividing factor(n-ddof)
        """
        mean = self.mean() if mean is None else mean
        # capsule, mean value, size, stride, ddof
        return libcuda.vector_std(self.data, mean, self.shape, 1, ddof)

    # meta methods
    def __init__(self, shape=1, source=None, dtype="float64", **kwds):
        # chain up
        super().__init__(**kwds)
        if source is not None:
            if isinstance(source, gsl.vector):
                self.shape=source.shape
                self.dtype = numpy.dtype(dtype)
                self.nbytes = self.dtype.itemsize*self.shape
                self.data = libcuda.vector_alloc(self.shape, self.nbytes, self.dtype.num)
                libcuda.vector_fromgsl(self.data, source.data)
            elif isinstance(source, numpy.ndarray):
                self.shape=source.size
                self.dtype = source.dtype
                self.nbytes = self.dtype.itemsize*self.shape
                self.data = libcuda.vector_alloc(self.shape, self.nbytes, self.dtype.num)
                libcuda.vector_fromnumpy(self.data, source)
            else:
                raise NotImplementedError("only gsl/numpy sources are supported")
        else:
            # adjust the shape, just in case
            shape = int(shape)
            # store
            self.shape = shape
            self.dtype = numpy.dtype(dtype)
            self.nbytes = self.dtype.itemsize*self.shape
            self.data = libcuda.vector_alloc(self.shape, self.nbytes, self.dtype.num)
        # all done
        return

    #properties
    @property
    def size(self):
        return self.shape

    # container support
    def __len__(self):
        # easy
        return self.shape

    # in-place arithmetic
    def __iadd__(self, other):
        """
        In-place addition with the elements of {other}
        """
        # if other is a vector
        if isinstance(other, Vector):
            libcuda.vector_iadd(self.data, other.data)
        # or a scalar
        elif isinstance(other, float) or isinstance(other, int):
            libcuda.vector_iadd_scalar(self.data, float(other))
        else:
            # otherwise, let the interpreter know
            raise NotImplemented
        return self

    def __isub__(self, other):
        """
        In-place subtraction with the elements of {other}
        """
        # if other is a vector
        if isinstance(other, Vector):
            libcuda.vector_isub(self.data, other.data)
        elif isinstance(other, float) or isinstance(other, int):
            val = -float(other)
            libcuda.vector_iadd_scalar(self.data, val)
        else:
            # otherwise, let the interpreter know
            raise NotImplemented
        return self

    def __imul__(self, other):
        """
        In-place scale with a factor {other}
        """
        # if other is a vector
        if isinstance(other, Vector):
            libcuda.vector_imul(self.data, other.data)
        # or a scalar
        elif isinstance(other, float) or isinstance(other, int):
            libcuda.vector_imul_scalar(self.data, float(other))
        else:
            # otherwise, let the interpreter know
            raise NotImplemented
        return self

    def __getitem__(self, index):
        """
        Get the value of v[index]
        :param index: index of the vector
        :return: float value (in cpu)
        """
        v = self.copy_to_host(type='numpy')
        return v[index]


    # private data
    data = None


# end of file
