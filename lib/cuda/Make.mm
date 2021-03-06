# -*- Makefile -*-
# -*- coding: utf-8 -*-
#
# (c) 1998-2019 all rights reserved
#
# Author(s): Lijun Zhu

# get the machinery for building shared objects
include shared/target.def

# get cuda
include cuda.def

# the project defaults
include pyre.def

# the package name
PACKAGE = cuda
# the sources
PROJ_SRCS = \
    timer.cu \
    elementwise.cu \
    linalg.cu \
    matrixops.cu \
    statistics.cu \

# the products
PROJ_SAR = $(BLD_LIBDIR)/libpyre$(PACKAGE).$(EXT_SAR)
PROJ_DLL = $(BLD_LIBDIR)/libpyre$(PACKAGE).$(EXT_SO)
# the private build space
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/lib/$(PACKAGE)
# what to clean
PROJ_CLEAN += $(EXPORT_LIBS) $(EXPORT_INCDIR)

PROJ_NVCC_FLAGS += -std=c++14
PROJ_LIBRARIES = -ljournal -lcublas -lcurand -lcusolver


# what to export
# the library
EXPORT_LIBS = $(PROJ_DLL)
# top level header
EXPORT_HEADERS = \
    cuda.h \
# headers scoped by the package name
EXPORT_PKG_HEADERS = \
    debug.h \
    error.h \
    helper.h \
    timer.h \
    cudalib.h \
    cuda_vector.h \
    cuda_matrix.h \
    elementwise.h \
    linalg.h \
    matrixops.h \
    atomic.h \
    reduction.h \
    statistics.h \

# standard targets
all: export

export:: $(PROJ_DLL) export-headers export-package-headers export-libraries

live: live-headers live-package-headers live-libraries

# archiving support
zipit:
	cd $(EXPORT_ROOT); \
	    zip -r $(PYRE_ZIP) lib/libpyre$(PACKAGE).$(EXT_SO); \
        zip -r $(PYRE_ZIP) ${addprefix include/pyre/, $(EXPORT_HEADERS)} ; \
        zip -r $(PYRE_ZIP) include/pyre/$(PACKAGE)

# end of file
