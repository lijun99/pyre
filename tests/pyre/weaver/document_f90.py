#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2012 all rights reserved
#


"""
Exercise a Fortran90 weaver
"""


def test():
    # get the package
    import pyre.weaver
    # instantiate a weaver
    weaver = pyre.weaver.newWeaver(name="sanity")
    weaver.language = "f90"

    text = list(weaver.weave())
    assert text == [
        '! -*- F90 -*-',
        '!',
        '! Michael A.G. Aïvázis',
        '! California Institute of Technology',
        '! (c) 1998-2012 All Rights Reserved',
        '!',
        '',
        '',
        '! end of file',
        ]

    return


# main
if __name__ == "__main__":
    test()


# end of file 
