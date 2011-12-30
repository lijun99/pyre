# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2012 all rights reserved
#


class Leaf:
    """
    Mix-in class that provides an implementation of the subset of the interface of {Node} that
    requires traversals of the expression graph rooted at leaf nodes.
    """


    # public data
    operators = [] # leaves have no dependencies
    operands = () # leaves have no operands


    @property
    def span(self):
        """
        Traverse my graph and yield all nodes in the graph
        """
        # just myself
        yield self
        # and no more
        return


    @property
    def variables(self):
        """
        Traverse my expression graph and return an iterable with all the variables in my graph

        Variables are reported as many times as they show up in my graph. Clients that are
        looking for the set unique dependencies have to prune the results themselves.
        """
        # just myself
        yield self
        # and no one else
        return


    # interface
    def substitute(self, current, replacement):
        """
        Traverse my expression graph and replace all occurrences of node {current} with
        {replacement}
        """
        # nothing to do
        return


    # meta methods
    def __init__(self, operands=(), **kwds):
        # swallow {operands} since leaves don't have any
        super().__init__(**kwds)
        return


# end of file 
