# -*- cmake -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


function(pyre_cudaPackage)
  # if the user requested CUDA support
  if(WITH_CUDA)
    # install the sources straight from the source directory
    install(
      DIRECTORY cuda
      DESTINATION ${PYRE_DEST_PACKAGES}
      FILES_MATCHING PATTERN *.py
      )
  endif()
  # all done
endfunction(pyre_cudaPackage)


# build libpyrecuda
function(pyre_cudaLib)
  # copy the cuda headers over to the staging area
  file(
    COPY cuda
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/pyre
    FILES_MATCHING
    PATTERN *.h PATTERN *.icc
    PATTERN cuda/cuda.h EXCLUDE
    )
  # and the cuda master header within the pyre directory
  file(
    COPY cuda/cuda.h
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/pyre
    )

  # the libpyrecuda target
  add_library(pyrecuda SHARED)
  # define the core macro
  set_target_properties(pyrecuda PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
  # set the include directories
  target_include_directories(
    pyrecuda PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    )
  # add the sources
  target_sources(pyrecuda
    PRIVATE
    cuda/elementwise.cu
    cuda/linalg.cu
    cuda/matrixops.cu
    cuda/statistics.cu
    cuda/timer.cu
    )

  # install libpyrecuda
  install(
    TARGETS pyrecuda
    LIBRARY
    )
  # all done
endfunction(pyre_cudaLib)

# build the cuda module
function(pyre_cudaModule)
  # the target
  Python3_add_library(cudamodule MODULE)
  # adjust the name to match what python expects
  set_target_properties(cudamodule PROPERTIES LINKER_LANGUAGE CUDA)
  set_target_properties(cudamodule PROPERTIES LIBRARY_OUTPUT_NAME cuda)
  set_target_properties(cudamodule PROPERTIES SUFFIX ${PYTHON3_SUFFIX})
  # set the include directories
  target_include_directories(cudamodule PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${Python3_NumPy_INCLUDE_DIRS})
  # set the libraries to link against
  set(CUDA_LIBRARIES cublas cusolver curand)
  target_link_libraries(
    cudamodule PRIVATE
    ${CUDA_LIBRARIES}
    ${GSL_LIBRARIES}
    pyre journal pyrecuda
    )
  # add the sources
  target_sources(cudamodule PRIVATE
    cuda/cuda.cc
    cuda/cublas.cc
    cuda/curand.cc
    cuda/cusolver.cc
    cuda/exceptions.cc
    cuda/device.cc
    cuda/discover.cc
    cuda/gsl.cc
    cuda/matrix.cc
    cuda/metadata.cc
    cuda/numpy.cc
    cuda/stream.cc
    cuda/timer.cc
    cuda/vector.cc
    cuda/stats.cc
    cuda/vector.cc
    )
  # copy the capsule definitions to the staging area
  file(
    COPY cuda/capsules.h
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../lib/pyre/cuda
    )
  if (${MPI_FOUND})
    # add the MPI aware sources to the pile
    target_sources(cudamodule PRIVATE cuda/mpi.cc)
    # the mpi include directories
    target_include_directories(cudamodule PRIVATE ${MPI_CXX_INCLUDE_PATH})
    # and the mpi libraries
    target_link_libraries(cudamodule PRIVATE ${MPI_CXX_LIBRARIES})
    # add the preprocessor macro
    target_compile_definitions(cudamodule PRIVATE WITH_MPI)
  endif()

  # install the extension
  install(
    TARGETS cudamodule
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/packages/cuda
    )
  # and publish the capsules
  install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/cuda/capsules.h ${CMAKE_CURRENT_SOURCE_DIR}/cuda/dtypes.h
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/pyre/cuda
    )
  # all done
endfunction(pyre_cudaModule)


# end of file
