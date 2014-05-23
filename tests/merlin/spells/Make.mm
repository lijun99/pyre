# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2014 all rights reserved
#


PROJECT = merlin-tests

TEST_DIR = /tmp

PROJ_CLEAN += \
    $(TEST_DIR)/merlin.deep \
    $(TEST_DIR)/merlin.one \
    $(TEST_DIR)/merlin.shallow \
    $(TEST_DIR)/merlin.two \

MERLIN = $(EXPORT_BINDIR)/merlin

#--------------------------------------------------------------------------
#

all: test

test: init clean

init:
	$(MERLIN) init $(TEST_DIR)/merlin.shallow
	$(MERLIN) init $(TEST_DIR)/merlin.one $(TEST_DIR)/merlin.two
	$(MERLIN) init --create-prefix $(TEST_DIR)/merlin.deep/ly/burried

# end of file 
