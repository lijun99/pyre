# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2020 all rights reserved
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

    # public data
    description = "cannot convert unit instance to float"

    # meta-methods
    def __init__(self, operand, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.op = operand
        # all done
        return


class CompatibilityError(UnitError):
    """
    Exception raised when the operands of a binary operator have imcompatible units such as
    adding lengths to times
    """

    # public data
    description = "{0.operation}: {0.op1} and {0.op2} are incompatible"

    # meta-methods
    def __init__(self, operation, op1, op2, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.operation = operation
        self.op1 = op1
        self.op2 = op2
        # all done
        return


# end of file
