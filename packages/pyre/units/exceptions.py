# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2012 all rights reserved
#


# exceptions
from ..framework.exceptions import FrameworkError


class UnitError(FrameworkError):
    """
    Base class for all errors generated by this package
    """


class ConversionError(UnitError):
    """
    Exception raised when an attempt was made to convert a dimensional quantity to a
    float. This typically happens when a math package function is invoked with a dimensional
    argument
    """

    def __init__(self, operand, **kwds):
        super().__init__(description="cannot convert unit instance to float", **kwds)
        self.op = operand
        return


class CompatibilityError(UnitError):
    """
    Exception raised when the operands of a binary operator have imcompatible units such as
    adding lengths to times
    """

    def __init__(self, operation, op1, op2, **kwds):
        msg = "{}: {} and {} are incompatible".format(operation, op1, op2)
        super().__init__(description=msg, **kwds)
        self.op1 = op1
        self.op2 = op2
        return


# end of file 
