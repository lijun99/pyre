# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2012 all rights reserved
#


import weakref
import collections


class Registrar:
    """
    The manager of interfaces, component class and their instances

    All user defined interfaces and component classes are registered with {Registrar} as they
    are encountered by the framework. Clients can discover the interface and component classes
    that are registered, the set of instances of any component, as well as the components that
    implement a particular interface.

    The two base classes {pyre.components.Interface} and {pyre.components.Component}, as well
    as the interface specifications autogenerated by {Role}, are declared with the special
    attribute {hidden} set to {True} and they are not registered.
    """


    # public state
    interfaces = None # the set of all registered interfaces
    components = None # a map of component classes to their instances
    implementors = None # a map of interfaces to component classes that implement them


    # interface
    def registerInterfaceClass(self, interface):
        """
        Register the {interface} class record
        """
        # add to the pile
        self.interfaces.add(interface)
        # and hand it back to the caller
        return interface
        

    def registerComponentClass(self, component):
        """
        Register the {component} class record

        This method is invoked by the {executive} when a new {component} declaration is
        encountered. 

        The {component} record is used to prime a weak map of component classes and their
        instances so the framework can keep track of the instantiated component. It is also
        added to the set of compatible implementations of all its known interfaces.  This
        enables the framework to answer questions about the possible implementations of a given
        interface.
        """
        # register it under its family name
        if component.pyre_family:
            # turn the family iterable into a single string
            family = component.pyre_getFamilyName()
            # and add it to the weak dictionary
            self.families[family] = component
        # prime the component extent
        self.components[component] = weakref.WeakSet()
        # update the map of interfaces it implements
        for interface in self.findRegisteredInterfaces(component):
            self.implementors[interface].add(component)
        # and hand it back to the caller
        return component


    def registerComponentInstance(self, component):
        """
        Register this {component} instance
        """
        # update the name map
        self.names[component.pyre_name] = component
        # add {component} to the set of registered instances of its class
        # Actor, the component metaclass, guarantees that component classes get registered
        # before any of their instances, so the lookup for the class should never fail
        try:
            self.components[type(component)].add(component)
        except KeyError:
            import journal
            firewall = journal.firewall("pyre.components")
            raise firewall.log(
                "pyre.components.Registrar: unregistered class {.__name__!r} "
                "of component {.pyre_name!r}".format(type(component), component))
        # and return the instance back to the caller
        return component


    # implementation details
    def findRegisteredInterfaces(self, component):
        """
        Build a sequence of the registered interfaces that are implemented by this component
        """
        # get the interface implementation specification
        implements = component.pyre_implements
        # bail out if there weren't any
        if implements is None:
            return
        # otherwise, loop over the interface mro
        for interface in implements.__mro__:
            # ignore the trivial interfaces by looking each one up in my registry
            if interface in self.interfaces:
                yield interface
        # all done
        return


    # meta methods
    def __init__(self, **kwds):
        super().__init__(**kwds)

        # the component registries
        self.components = {} # map: component class -> set of instances
        self.interfaces = set() # a collection of known interfaces
        self.implementors = collections.defaultdict(set) # map: interfaces -> components

        self.names = weakref.WeakValueDictionary() # map: component names -> component instances
        self.families = {} # map: component families -> component classes

        return


# end of file 
