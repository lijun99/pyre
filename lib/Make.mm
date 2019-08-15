# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# my subdirectories
RECURSE_DIRS = \
    journal \
    pyre \

# the optional packages
# mpi
ifneq ($(strip $(MPI_DIR)),)
  RECURSE_DIRS += mpi
endif

# cuda
ifneq ($(strip $(CUDA_DIR)),)
  RECURSE_DIRS += cuda
endif

# use a tmp directory that knows what we are building in this directory structure
PROJ_TMPDIR = $(BLD_TMPDIR)/lib

# the standard targets
all: export


tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

export:: export-portinfo
	BLD_ACTION="export" $(MM) recurse

live:
	BLD_ACTION="live" $(MM) recurse

# archiving support
zipit:
	PYRE_ZIP=$(PYRE_ZIP) BLD_ACTION="zipit" $(MM) recurse

# exporting the portinfo settings
export-portinfo: $(EXPORT_ROOT)/include/portinfo


$(EXPORT_ROOT)/include/portinfo: $(EXPORT_ROOT)/include Make.mm
	@sed \
          -e "s:PYRE_PLATFORM:#define mm_platforms_${MM_PLATFORM}_${MM_ARCH} 1:g" \
          portinfo > $@

# end of file
