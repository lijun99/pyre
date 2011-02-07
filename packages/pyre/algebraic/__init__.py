# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2011 all rights reserved
#


"""
This package provides the machinery for implementing deferred evaluation in python.

The strategy is to capture arithmetic operations among a collection of instances as an
expression tree built out of the operators in this package. Actual evaluation takes place when
the {eval} method is called.

The classes in this package provide a complete set of overloaded special methods for all the
arithmetic operations. All you have to do is derive from {Node} for all classes for which you
want to enable this behavior, and implement {eval} so the expression can be evaluated.
"""


# end of file 
