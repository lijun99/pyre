# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2012 all rights reserved
#


import pyre

class Functor(pyre.interface):
    """
    Facility declaration for function objects
    """

    # interface
    @pyre.provides
    def eval(self, points):
        """
        Evaluate the function at the supplied points
        """


# end of file 
