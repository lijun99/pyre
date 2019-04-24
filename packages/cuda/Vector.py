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
    Cuda vector
    """

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

    def sum(self):
        """
        summation
        """
        host = self.copy_to_host(type='numpy')
        return host.sum()


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
            return self
        # otherwise, let the interpreter know
        raise NotImplemented
        
    def __isub__(self, other):
        """
        In-place subtraction with the elements of {other}
        """
        # if other is a vector
        if isinstance(other, Vector):
            libcuda.vector_isub(self.data, other.data)
            return self
        # otherwise, let the interpreter know
        raise NotImplemented

    def __imul__(self, other):
        """
        In-place scale with a factor {other}
        """
        # if other is a matrix
        if isinstance(other, float) or isinstance(other, int):
            libcuda.vector_imul(self.data, float(other))
            return self
        # otherwise, let the interpreter know
        raise NotImplemented

    # private data
    data = None


# end of file
