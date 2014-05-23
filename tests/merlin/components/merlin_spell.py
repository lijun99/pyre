#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2014 all rights reserved
#


"""
Verify that the spellbook can locate spells
"""


def test():
    # access to the merlin executive
    from merlin import merlin
    # get the spellbook
    spellbook = merlin.spellbook

    # ask it to find a spell
    spell = spellbook.find(name="sample")

    # and return
    return spell


# main
if __name__ == "__main__":
    test()


# end of file 
