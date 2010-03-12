#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2010 all rights reserved
#


"""
Verify that reference nodes correctly reflect the value of their referends
"""


import pyre.calc


def test():

    # make a node and set its value
    v = 80.
    production = pyre.calc.newNode(value=v)
    clone = production.newReference()

    assert production.value == v
    assert clone.value == v
    
    # once more
    v = 100.
    production.value = v
    assert production.value == v
    assert clone.value == v

    return


# main
if __name__ == "__main__":
    # get the extent manager
    from pyre.patterns.ExtentAware import ExtentAware
    # install it
    pyre.calc._metaclass_Node = pyre.calc._metaclass_Evaluator = ExtentAware
    # run the test
    test()
    # verify reference counts
    # for nodes
    from pyre.calc.Node import Node
    # print([node for node in Node._pyre_extent])
    assert set(Node._pyre_extent) == set()
    # for evaluators
    from pyre.calc.Evaluator import Evaluator
    # print([evaluator for evaluator in Evaluator._pyre_extent])
    assert set(Evaluator._pyre_extent) == set()


# end of file 
