# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 1998-2026 all rights reserved


# my superclass
from ..Codec import Codec


# class declaration
class Config(Codec):
    """
    This package contains the implementation of the {pfg} reader and writer
    """

    # constants
    encoding = "pfg"

    # interface
    @classmethod
    def decode(cls, uri, source, locator):
        """
        Parse {source} and return the configuration events it contains
        """
        # get the parser factory
        from .Parser import Parser

        # make a parser
        parser = Parser()
        # harvest the configuration events; the parser is lazy, so drive it to the end
        configuration = list(parser.parse(uri=uri, stream=source, locator=locator))
        # grab the accumulated errors
        errors = parser.errors
        # if there were no errors
        if not errors:
            # return the harvested configuration events
            return configuration
        # otherwise, list the errors with their locations, escaped for the description template
        description = "\n".join(map(str, errors)).replace("{", "{{").replace("}", "}}")
        # and complain
        raise cls.DecodingError(codec=cls, uri=uri, description=description)


# end of file
