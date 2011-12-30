#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2012 all rights reserved
#


"""
Verify that node insertion fails when an intermediate path component is not a folder
"""


def test():
    import pyre.filesystem

    # build a folder
    root = pyre.filesystem.virtual()
    # and a node
    mga = root.node()
    # add it to the folder
    root._insert(uri="/home/users/mga", node=mga)
    # now create another node
    tmp = root.node()
    # and attempt to add it to mga
    try:
        root._insert(uri="/home/users/mga/tmp", node=tmp)
        assert False
    except root.FolderInsertionError as error:
        assert (
            error.description 
            == "error while inserting '/home/users/mga/tmp': 'mga' is not a folder")

    # all done
    return root


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file 
