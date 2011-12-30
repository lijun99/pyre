#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2012 all rights reserved
#


"""
Verify that expressions work
"""


def test():
    import pyre.algebraic

    # set up the model
    model = pyre.algebraic.model(name="expression")

    # the nodes
    p = 80.
    s = .25*80
    # register the nodes
    model["production"] = p
    model["shipping"] = s
    model["cost"] = "{production}+{shipping}"
    model["price"] = "2*{cost}"

    # check the values
    # print("before:")
    # print("  production:", model["production"])
    # print("  shipping:", model["shipping"])
    # print("  cost:", model["cost"])
    # print("  price:", model["price"])
    assert model["production"] == p
    assert model["shipping"] == s
    assert model["cost"] == p+s
    assert model["price"] == 2*(p+s)

    # make a change
    p = 100.
    model["production"] = p

    # check again
    # print("after:")
    # print("  production:", model["production"])
    # print("  shipping:", model["shipping"])
    # print("  cost:", model["cost"])
    # print("  price:", model["price"])
    assert model["production"] == p
    assert model["shipping"] == s
    assert model["cost"] == p+s
    assert model["price"] == 2*(p+s)

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()


# end of file 
