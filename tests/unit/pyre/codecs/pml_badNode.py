#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2010 all rights reserved
#


"""
Sanity check: verify that the package is accessible
"""


def test():
    # package access
    import pyre.codecs
    import pyre.config
    # get the codec manager
    m = pyre.codecs.newManager()
    # ask for a pml codec
    reader = m.newCodec(encoding="pml")
    # open a stream with an error
    sample = open("sample-badNode.pml")
    # read the contents
    try:
        reader.decode(configurator=None, stream=sample)
        assert False
    except reader.DecodingError as error:
        assert str(error) == (
            "decoding error in file='sample-badNode.pml', line=12, column=77: mismatched tag"
            )
 
    return m, reader


# main
if __name__ == "__main__":
    test()


# end of file 
