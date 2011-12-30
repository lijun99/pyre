#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2012 all rights reserved
#


"""
Verify that info channels by the same name share a common state
"""


def test():
    # access the package
    import journal
    # build a info channel
    info = journal.info("activation")
    # verify that it is on by default, activated from a configuration source
    assert info.active == True
    # disable it
    info.active = False

    # access the same channel through another object
    clone = journal.info("activation")
    # verify that it is off 
    assert clone.active == False
    # enable it
    clone.active = True

    # check that the other channel has been activated as well
    assert info.active == True

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file 
