#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2010 all rights reserved
#


"""
Exercise the simple filesystem explorer
"""


def test():
    import pyre.filesystem

    # make a filesystem
    fs = pyre.filesystem.newVirtualFilesystem()
    # create a couple of nodes and insert them into the filesystem
    fs.insert(node=fs.newNode(), path="/home/users/mga/dv/tools/bin/hello")
    fs.insert(node=fs.newNode(), path="/home/users/mga/dv/tools/lib/libhello.a")

    # explore
    explorer = pyre.filesystem.newTreeExplorer()
    contents = explorer.explore(fs)

    # check
    assert contents == [
        " / (d)",
        " `- home (d)",
        "    `- users (d)",
        "       `- mga (d)",
        "          `- dv (d)",
        "             `- tools (d)",
        "                +- bin (d)",
        "                |  `- hello (f)",
        "                `- lib (d)",
        "                   `- libhello.a (f)",
        ]

    return


# main
if __name__ == "__main__":
    test()


# end of file 
