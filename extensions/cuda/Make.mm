# -*- Makefile -*-
#
# michael a.g. aïvázis, orthologue
# lijun zhu, caltech
# (c) 1998-2019 all rights reserved
#

# project defaults
include cuda.def
# package name
PACKAGE =
# the module
MODULE = cuda
# get the cuda support
include cuda/default.def
# get gsl support
include gsl/default.def
# get numpy support
include numpy/default.def
# and build a python module
include std-pythonmodule.def
# use a tmp directory that knows the name of the module
PROJ_TMPDIR = $(BLD_TMPDIR)/extensions/$(PROJECT)
# link against these
PROJ_LIBRARIES = -ljournal -lpyrecuda -lcurand -lcublas -lcusolver
# the sources
PROJ_SRCS = \
    device.cc \
    discover.cc \
    exceptions.cc \
    metadata.cc \
    vector.cc \
    matrix.cc \
    gsl.cc \
    numpy.cc \
    curand.cc \
    cublas.cc \
    cusolver.cc \
    stream.cc \
    timer.cc \

# actions
export:: export-headers

EXPORT_INCDIR = $(EXPORT_ROOT)/include/pyre/$(PROJECT)
EXPORT_HEADERS = \
    cudaext.h \
    capsules.h \
    dtypes.h \

# end of file
