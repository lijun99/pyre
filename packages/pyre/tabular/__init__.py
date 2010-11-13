# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2010 all rights reserved
#


# access to the basic objects in the package
from .Sheet import Sheet as sheet
from .Chart import Chart as chart
from .Pivot import Pivot as pivot

from .Record import Record as record
from .Measure import Measure as measure
from .Derivation import Derivation as derivation

# readers/writers
from .CSV import CSV as csv


# access the type declarators
from .. import schema


# convenience factories that build measures of specific types
def dimensional(default=0, **kwds):
    """
    Build a measure that has units

    Legal assignments are constrained to have units compatible with the default value
    """
    m = measure(**kwds)
    m.type = schema.dimensional
    m.default = default
    return m


def float(default=0, **kwds):
    """
    Build a float measure
    """
    m = measure(**kwds)
    m.type = schema.float
    m.default = default
    return m


def int(default=0, **kwds):
    """
    Build an integer measure
    """
    m = measure(**kwds)
    m.type = schema.int
    m.default = default
    return m


def str(default="", **kwds):
    """
    Build a string measure
    """
    m = measure(**kwds)
    m.type = schema.str
    m.default = default
    return m


# dimension factories
def inferred(measure, **kwds):
    """
    Build a dimension that assumes values in the range of {measure}
    """
    from .InferredDimension import InferredDimension
    return InferredDimension(measure=measure, **kwds)


# end of file 
