# -*- Makefile -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 1998-2020 all rights reserved


# project meta-data
merlin.major := $(repo.major)
merlin.minor := $(repo.minor)
merlin.micro := $(repo.micro)
merlin.revision := $(repo.revision)


# merlin builds a python package
merlin.packages := merlin.pkg
# no library
merlin.libraries :=
# no python extension
merlin.extensions :=
# test suite
merlin.tests := merlin.pkg.tests


# the merlin package meta-data
merlin.pkg.root := packages/merlin/
merlin.pkg.stem := merlin
merlin.pkg.ext :=


# get the testsuites
include $(merlin.tests)


# end of file