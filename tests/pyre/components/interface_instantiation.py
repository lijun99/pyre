#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2012 all rights reserved
#


"""
Verify that an exception gets raised when an interface is instantiated
"""


def test():
    import pyre

    # shut the firewall up
    import journal
    journal.firewall("pyre.components").active = False

    # declare
    class interface(pyre.interface):
        """a trivial interface"""
        p = pyre.property()

    # attempt to instantiate
    try:
        interface()
        assert False
    except journal.FirewallError:
        pass

    return interface


# main
if __name__ == "__main__":
    test()


# end of file 
