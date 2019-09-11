#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Shuffle a vector
"""


def test():
    # package access
    import gsl
    # make a vector and initialize it
    v = gsl.vector(shape=100)
    # prime
    for index in range(v.shape): v[index] = 2*index+1
    # shuffle
    vs = v.shuffle(rng=gsl.rng())
    # check it
    assert vs.mean() == v.mean()
    assert vs.max() == v.max()
    assert vs.min() == v.min()
    # all done
    return v


# main
if __name__ == "__main__":
    test()


# end of file
