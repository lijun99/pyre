# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2011 all rights reserved
#


# we build weak references to nodes
import weakref


# class declaration
class Memo:
    """
    A mix-in class that implements value memoization
    """


    # interface
    def getValue(self):
        """
        Override the node value retriever and return the contents of my value cache if it is up
        to date; otherwise, recompute the value and update the cache
        """
        # if my cache is invalid
        if self._value is None:
            # recompute
            self._value = super().getValue()
        # return the cache contents
        return self._value


    def setValue(self, value, **kwds):
        """
        Override the value setter to invalidate my cache and notify my observers
        """
        # update the value
        super().setValue(value=value, **kwds)
        # invalidate my cache and notify my observers
        self.notifyObservers()
        # and return
        return


    # cache management
    def pyre_flush(self, node=None):
        """
        Invalidate my cache and notify my observers
        """
        # do nothing if my cache is already invalid
        if self._value is None: return
        # invalidate the cache
        self._value = None
        # notify my observers
        return self.notifyObservers()
        

    def notifyObservers(self):
        """
        Notify the nodes that depend on me that my value has changed
        """
        # initialize the list of dead references
        dead = []
        # notify my observers
        for oref in self.observers:
            # get the node
            observer = oref()
            # if it is still alive
            if observer is not None:
                # flush it
                observer.pyre_flush(node=self)
            # otherwise
            else:
                # put its reference on the discard pile
                dead.append(oref)
        # clean up
        for ref in dead: self.observers.remove(ref)
        # and return
        return


    # observer management
    def subsume(self, obsolete):
        """
        Remove {obsolete} from its upstream graph and assume its responsibilities
        """
        # iterate over the observers of the {obsolete} node
        for oref in tuple(obsolete.observers):
            # get the actual node
            observer = oref()
            # skip dead nodes
            if observer is None: continue
            # ask the observer to replace {obsolete} with me
            observer.pyre_substitute(current=obsolete, replacement=self)
        # all done
        return
        

    def addObserver(self, node):
        """
        Add {node} to the set of nodes that depend on my value
        """
        # build a weak reference to {node} and add it to the pile
        self.observers.add(weakref.ref(node))
        # all done
        return


    def removeObserver(self, node):
        """
        Remove {node} from the set of nodes that depend on my value
        """
        # build a weak reference to {node} and remove it from the pile
        self.observers.remove(weakref.ref(node))
        # all done
        return


    # meta methods
    def __init__(self, operands=(), **kwds):
        super().__init__(operands=operands, **kwds)

        # initialize the set of my observers
        self.observers = set()
        # add me as an observer to each of my operands
        for operand in operands: operand.observers.add(weakref.ref(self))

        # and return
        return


    # implementation details
    def _substitute(self, index, current, replacement):
        """
        Adjust the operands by substituting {replacement} for {current} in the list of operands
        at position {index}
        """
        # flush my cache
        self.pyre_flush()
        # make a weak reference to myself
        selfref = weakref.ref(self)
        # remove me as an observer of the old node
        current.observers.remove(selfref)
        # and add me to the list of observers of the replacement
        replacement.observers.add(selfref)
        # and ask my superclass to do the rest
        return super()._substitute(index, current, replacement)


    # private data
    _value = None


# end of file 
