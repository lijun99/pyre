# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include cuda.def
# package name
PACKAGE = cuda
# my headers live here
PROJ_INCDIR = $(BLD_INCDIR)/pyre/$(PROJECT)
# the python modules
EXPORT_PYTHON_MODULES = \
    Device.py \
    DeviceManager.py \
    exceptions.py \
    Timer.py \
    Vector.py \
    Matrix.py \
    cuRand.py \
    cuBlas.py \
    cuSolverDn.py \
    Stats.py \
    __init__.py

# standard targets
all: export

export:: export-python-modules

live: live-python-modules

# end of file
