# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2012 all rights reserved
#


# declaration
class Shelf:
    """
    A symbol table built on top of a native python module
    """


    # exceptions
    from ..exceptions import SymbolNotFoundError


    # public data
    module = None

    @property
    def source(self):
        """
        Return the filename associated with this module
        """
        return self.module.__file__


    # interface
    def retrieveSymbol(self, symbol):
        """
        Retrieve {symbol} from this shelf
        """
        try:
            return getattr(self.module, symbol)
        except AttributeError as error:
            raise self.SymbolNotFoundError(shelf=self, symbol=symbol) from error


    # meta methods
    def __init__(self, module, locator,  **kwds):
        super().__init__(**kwds)
        self.module = module
        self.locator = locator
        return


# end of file 
