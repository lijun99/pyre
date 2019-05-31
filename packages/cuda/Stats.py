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
from .Vector import Vector as vector
from .Matrix import Matrix as matrix

# the class declaration
class Stats:
    """
    Statistics Routines for cuda vectors ana matrices
    """

    def amin(a):
        """
        Return the minimum of a vector or a matrix or minimum along an axis.
        :param a: a cuda vector or matrix
        :return: the minimum value
        """

        if isinstance(a, vector):
            out = libcuda.vector_amin(a.data, a.shape, 1)
        elif isinstance(a, matrix):
            out = libcuda.matrix_amin(a.data, a.size, 1)
        else:
            raise NotImplementedError("unsupported type {}".format(type(a)))

        # all done
        return out

    def amax(a):
        """
        Return the maximum of a vector or a matrix or maximum along an axis.
        :param a: a cuda vector or matrix
        :return: the maximum value
        """
        if isinstance(a, vector):
            out = libcuda.vector_amax(a.data, a.shape, 1)
        elif isinstance(a, matrix):
            out = libcuda.matrix_amax(a.data, a.size, 1)
        else:
            raise NotImplementedError("unsupported type {}".format(type(a)))

        # all done
        return out

    def l1norm(x, size=None, stride=1):
        """
        L1 norm of a vector \sum_i |x_i|
        :return:
        """
        if isinstance(x, vector):
            size = x.shape if size is None else size
            out = libcuda.L1norm(x.data, size, stride)
        else:
            raise NotImplementedError("unsupported type {}".format(type(x)))

        return out

    def l2norm(x, size=None, stride=1):
        """
        L2 norm of a vector \sqrt{\sum_i x_i^2}
        :return:
        """
        if isinstance(x, vector):
            size = x.shape if size is None else size
            out = libcuda.L2norm(x.data, size, stride)
        else:
            raise NotImplementedError("unsupported type {}".format(type(x)))

        return out

    def linfnorm(x, size=None, stride=1):
        """
        L-Infinity norm of a vector max|x_i|
        :return:
        """
        if isinstance(x, vector):
            size = x.shape if size is None else size
            out = libcuda.Linfnorm(x.data, size, stride)
        else:
            raise NotImplementedError("unsupported type {}".format(type(x)))

        return out


    def covariance(x, y, out=None, axis=0, ddof=1):
        """
        Compute covariance between two vectors or matrices (along row or col)
        cov(X,Y) = E[(X-E[X])(Y-E[Y])]
        :param x, y: two input vectors or matrices
        :param axis: (for matrices only) 0/1 = along row/col
        :param out: (for matrices only) vector of size shape[1]/shape[0] for axis = 0/1
        :return: float for vectors, vector for matrices
        """

        # check the input type
        # vectors
        if isinstance(x, vector) and isinstance (y, vector):
            out = libcuda.vector_covariance(x.data, y.data, x.size, 1, ddof)
        elif isinstance(x, matrix) and isinstance(y, matrix):
            # allocate output vector if not available
            if axis ==0 : # along row
                out = out if out is not None else vector(shape=x.shape[1], dtype=x.dtype)
            else:
                out = out if out is not None else vector(shape=x.shape[0], dtype=x.dtype)
            # call correlation
            libcuda.matrix_covariance(x.data, y.data, out.data, axis, ddof)
        else:
            raise NotImplementedError("unsupported type {}".format(type(x)))

        return out

    def correlation(x, y, axis=0, out=None):
        """
        Compute correlation between two vectors or matrices (along row or col)
        cor(X,Y) = cov(X, Y)/ (std(X)std(Y))
        :param y:
        :param axis:
        :param out:
        :return:
        """

        if isinstance(x, vector) and isinstance (y, vector):
            out = libcuda.vector_correlation(x.data, y.data, x.size, 1)
        elif isinstance(x, matrix) and isinstance(y, matrix):
            # allocate output vector if not available
            if axis ==0 : # along row
                out = out if out is not None else vector(shape=x.shape[1], dtype=x.dtype)
            else:
                out = out if out is not None else vector(shape=x.shape[0], dtype=x.dtype)
            # call correlation
            libcuda.matrix_correlation(x.data, y.data, out.data, axis)
        else:
            raise NotImplementedError("unsupported type {}".format(type(x)))

        return out

    def max_diff(x, y):
        """
        compute maximum difference between elements of two vectors or matrices
        max{|x_i - y_i|}
        :param x, y: two vectors or matrices
        :return: the maximum difference max|x_i - y_i|
        """
        # vectors
        if isinstance(x, vector) and isinstance (y, vector):
            # make a copy of x
            diff = x.clone()
            # subtract from y
            diff -= y
            # find the max diff using L-Infinity norm
            out = libcuda.Linfnorm(diff.data, diff.size, 1)
        elif isinstance(x, matrix) and isinstance(y, matrix):
            diff = x.clone()
            diff -= y
            diff = diff.tovector()
            out = libcuda.Linfnorm(diff.data, diff.size, 1)
        else:
            raise NotImplementedError("unsupported type {}".format(type(x)))

        return out


# end of file
