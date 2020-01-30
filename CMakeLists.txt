# -*- cmake -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# cmake setup
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
# policies
if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)
cmake_policy(SET CMP0060 NEW)
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif(POLICY CMP0074)
if (POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif(POLICY CMP0076)

# options
option(WITH_CUDA "enable support for CUDA" ON)

# adjust the include path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/.cmake)
# get support
include(pyre_init)
include(pyre_journal)
include(pyre_pyre)
include(pyre_merlin)
include(pyre_gsl)
include(pyre_mpi)
include(pyre_cuda)

# ask git for the pyre version
pyre_getVersion()

# set up the project
project(PYRE VERSION ${REPO_MAJOR}.${REPO_MINOR}.${REPO_MICRO} LANGUAGES CXX)

# packages
# python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
# gsl
find_package(GSL)
# mpi
find_package(MPI)
# postgres
find_package(PostgreSQL)

# set up cmake
pyre_cmakeInit()
# set up c++
pyre_cxxInit()
# set up python
pyre_pythonInit()
# set up cuda
if (WITH_CUDA)
  # check whether cuda toolkit is available
  include(CheckLanguage)
  check_language(CUDA)
  # available
  if(CMAKE_CUDA_COMPILER)
    # turn it on
    enable_language(CUDA)
    message(STATUS "CUDA Toolkit found and CUDA support is enabled")
    # other cuda settings
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 11)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
  # cuda toolkit not available
  else()
    # turn off WITH_CUDA option
    message(STATUS "CUDA Toolkit not found and set WITH_CUDA to OFF")
    set(WITH_CUDA OFF)
  endif()
endif()

# initialize the variables that describe the staging directory layout
pyre_stagingInit()
# initialize the variables that describe the install directory layout
pyre_destinationInit()

# visit subdirectories
add_subdirectory(packages)
add_subdirectory(lib)
add_subdirectory(extensions)
add_subdirectory(defaults)
add_subdirectory(bin)

# end of file
