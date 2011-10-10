# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2011 all rights reserved
#


# packages
import re
import pyre.patterns
# super-class
from .SymbolTable import SymbolTable


# declaration
class Hierarchical(SymbolTable):
    """
    Storage and naming services for algebraic nodes

    This class assumes that the node names form a hierarchy, very much like path names. They
    are expected to be given as tuples of strings that specify the names of the "folders" at
    each level.

    HierarchicalModel provides support for links, entries that are alternate names for other
    folders.
    """


    # types
    # my node type
    from .Node import Node


    # public data
    separator = '.'


    # interface
    # model traversal
    def select(self, pattern=''):
        """
        Generate a sequence of (name, value) pairs for all nodes in the model whose name
        matches the supplied {pattern}. Careful to properly escape periods and other characters
        that may occur in the name of the requested keys that are recognized by the {re}
        package
        """
        # check whether i have any nodes
        if not self._nodes: return
        # build the name recognizer
        regex = re.compile(pattern)
        # iterate over my nodes
        for node in self.nodes:
            # get the name of the node
            name = node.name
            # if the name matches
            if regex.match(name):
                # yield the name and the node
                yield name, node
        # all done
        return


    def children(self, key):
        """
        Given the address {key} of a node, iterate over all the canonical nodes that are
        its logical children
        """
        # hash the root key
        # print("HierarchicalModel.children: key={}".format(key))
        hashkey = self._hash.hash(key)
        # print("   names: {}".format(key.nodes.items()))
        # extract the unique hash subkeys
        unique = set(hashkey.nodes.values())
        # iterate over the unique keys
        for key in unique:
            # print("  looking for:", key)
            # extract the node
            try:
                node = self._nodes[key]
            # if not there...
            except KeyError:
                # it's because the key exists in the model but none of its immediate children
                # are leaf nodes with associated values. this happens often for configuration
                # settings to facilities that have not yet been converted into concrete
                # components; it also happens for configuration settings that are not meant for
                # components at all, such as journal channel activations.
                continue
            # extract the required information
            yield key, node

        # all done
        return


    # alternative node access
    def alias(self, *, alias, canonical):
        """
        Register the name {alias} as an alternate name for {canonical}
        """
        # build the multikeys
        aliasKey = alias.split(self.separator)
        canonicalKey = canonical.split(self.separator)
        # ask the hash to alias the two names and retrieve the corresponding hash keys
        aliasHash, canonicalHash = self._hash.alias(alias=aliasKey, canonical=canonicalKey)

        # now that the two names are aliases of each other, we must resolve the potential node
        # conflict: only one of these is accessible by name any more

        # look for a preëxisting node under the alias
        try:
            aliasNode = self._nodes[aliasHash]
        # if the lookup fails
        except KeyError:
            # no node has been previously registered under this alias, so we are done. if a
            # registration appears, it will be treated as a duplicate and patched appropriately
            return self
        # now, look for the canonical node
        try:
            canonicalNode = self._nodes[canonicalHash]
        # if there was no canonical node
        except KeyError:
            # install the alias as the canonical 
            self._nodes[canonicalHash] = aliasNode
            # all done
            return
        # either way clean up after the obsolete aliased node
        finally:
            # nothing could hash to {aliasHash} any more, so clear out the entry
            del self._nodes[aliasHash]

        # if we get this far, both preëxisted; the aliased info has been cleared out, the
        # canonical is as it should be. all that remains is to patch the two nodes
        self._update(identifier=canonicalHash, existing=aliasNode, replacement=canonicalNode)

        # all done
        return self
        

    # meta methods
    def __init__(self, **kwds):
        super().__init__(**kwds)

        # name hashing algorithm storage strategy
        self._hash = pyre.patterns.newPathHash()

        return


    # implementation details
    def _resolve(self, name):
        """
        Find the named node
        """
        # find and return the node and its identifier
        return self._retrieveNode(key=name.split(self.separator), name=name)


    def _retrieveNode(self, key, name):
        """
        Retrieve the node associated with {name}
        """
        # hash it
        hashkey = self._hash.hash(key)
        # attempt
        try:
            # to retrieve and return the node
            return self._nodes[hashkey], hashkey
        # if not there
        except KeyError:
            # no worries
            pass

        # build the name
        name = self.separator.join(key)
        # create a new node
        node = self._buildPlaceholder(name=name, identifier=hashkey if key else None)
        # if the request happened with a valid key
        if key:
            # register the new node
            self._nodes[hashkey] = node
        # and return the node and its identifier
        return node, hashkey


    # debug support
    def dump(self, pattern=''):
        """
        List my contents
        """
        print("model {0!r}:".format(self.name))
        print("  nodes:")
        for name, node in self.select(pattern):
            try: 
                value = node.value
            except self.UnresolvedNodeError:
                value = "unresolved"
            print("    {!r} <- {!r}".format(name, value))
        return


# end of file 
