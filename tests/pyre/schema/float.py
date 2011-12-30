#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2012 all rights reserved
#


"""
Check that float conversions work as expected
"""


def test():
    import pyre.schema

    # create a descriptor
    descriptor = pyre.schema.float

    # casts
    # successful
    assert 1.2 == descriptor.pyre_cast(1.2)
    assert 1.2 == descriptor.pyre_cast("1.2")
    # failures
    try:
        descriptor.pyre_cast(test)
        assert False
    except descriptor.CastingError as error:
        assert str(error) == "float() argument must be a string or a number"
        
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file 
