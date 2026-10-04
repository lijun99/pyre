#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 1998-2026 all rights reserved


"""
Verify that a configuration file named on the command line that has a syntax error is an error
"""


def test():
    # support
    import journal
    import pyre

    # the report is not what is being tested here
    journal.error("pyre.config").device = journal.trash()

    # get the executive instance
    executive = pyre.executive
    # and build a command line parser
    parser = executive.newCommandLineParser()
    # build an argument list that names a file with a syntax error
    commandline = [
        "--config=sample-badToken.pfg",
    ]
    # attempt to
    try:
        # parse it, which loads the configuration files it names
        parser.parse(commandline)
    # which must complain
    except journal.ApplicationError:
        # as it should
        pass
    # anything else is a failure
    else:
        assert False, "a malformed configuration file was ignored"

    # all done
    return parser


# main
if __name__ == "__main__":
    # do...
    test()


# end of file
