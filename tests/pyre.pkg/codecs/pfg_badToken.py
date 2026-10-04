#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 1998-2026 all rights reserved


"""
Verify that a pfg file with a syntax error is refused, with the location of the error
"""


def test():
    # package access
    import pyre.config

    # get the codec manager
    m = pyre.config.newConfigurator()
    # ask for a pfg codec
    reader = m.codec(encoding="pfg")
    # the configuration file
    uri = "sample-badToken.pfg"
    # open a stream with an error
    sample = open(uri)
    # read the contents
    try:
        # which must fail
        reader.decode(uri=uri, source=sample, locator=None)
        # so getting here is an error
        assert False
    # with a decoding error
    except reader.DecodingError as error:
        # that points at the offending token
        assert str(error) == (
            "file='sample-badToken.pfg', line=10, column=14: could not match 'michael\\n'"
        )

    # all done
    return m, reader


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
