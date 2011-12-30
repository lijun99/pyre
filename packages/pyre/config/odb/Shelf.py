# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2012 all rights reserved
#


class Shelf(dict):
    """
    Shelves are symbol tables that map component record factories to their names.

    Consider a configuration event, such as the command line instruction

        --integrator=vfs:/gauss/integrators.py/montecarlo

    This causes the manager of the persistent store to attempt to locate a file with the
    logical address "gauss/integrators.py". If the file exists, it is parsed and all the
    symbols it defines are loaded into a Shelf, with the names of the symbols as keys and the
    corresponding python objects as the values. Note that in our example, "montecarlo" is
    expected to be one of these symbols, and it is further expected that it is a callable that
    returns the class record of a component that is assignment compatible with the facility
    "integrator", but that is handled by the configuration manager and does not concern the
    shelf, which has been loaded successfully.

    The framework guarantees that each configuration file is loaded into one and only one
    shelf, and that this happens no more than once. This ensures that each component class
    record gets a unique id in the application process space, or that processing instructions
    in configuration files are executed only the first time the configuration file is loaded.
    """


    # exceptions
    from ..exceptions import SymbolNotFoundError


    # interface
    def retrieveSymbol(self, symbol):
        """
        Retrieve {symbol} from this shelf
        """
        try:
            return self[symbol]
        except KeyError as error:
            raise self.SymbolNotFoundError(shelf=self, symbol=symbol) from error


    # meta methods
    def __init__(self, locator, **kwds):
        super().__init__(**kwds)
        self.locator = locator
        return


# end of file 
