# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
from . import cuda as libcuda

class Device:
    """
    The property sheet of a CUDA capable device
    """


    # attributes
    id = None
    name = ""

    capability = ()
    driverVersion = ()
    runtimeVersion = ()
    computeMode = 0

    managedMemory = False
    unifiedAddressing = False

    processors = 0
    coresPerProcessor = 0

    globalMemory = 0
    constantMemory = 0
    sharedMemoryPerBlock = 0

    warp = 0
    maxThreadsPerBlock = 0
    maxThreadsPerProcessor = 0

    maxGrid = ()
    maxThreadBlock = ()

    _cublas_handle = None
    _curand_generator = None
    _cusolverdn_handle = None

    def initialize(self):
        """
        Initialize device and its handles
        """
        libcuda.setDevice(self.id)
        #if self.cublas_handle is None:
        #    self.cublas_handle = cublas.create_handle()
        #if self.curand_generator is None:
        #    self.curand_generator = curand.create_generator()
        return self

    @property
    def cublas_handle(self):
        """
        Return (create if not exist) a cublas handle
        :return:
        """
        from .cuBlas import cuBlas as cublas
        if self._cublas_handle is None:
            self._cublas_handle = cublas.create_handle()
        return self._cublas_handle

    def get_cublas_handle(self):
        return self.cublas_handle

    @property
    def cusolverdn_handle(self):
        from .cuSolverDn import cuSolverDn as cusolver
        if self._cusolverdn_handle is None:
            self._cusolverdn_handle = cusolver.create_handle()
        return self._cusolverdn_handle

    def get_cusolverdn_handle(self):
        return self.cusolverdn_handle

    @property
    def curand_generator(self):
        """
        Get (create if not exist) the curand_generator attached to device
        :return:
        """
        from .cuRand import cuRand as curand
        if self._curand_generator is None:
            self._curand_generator = curand.create_generator()
        return self._curand_generator

    def get_curand_generator(self):
        return self.curand_generator

    def reset(self):
        """
        Reset the current device
        """
        # easy enough
        return libcuda.resetDevice()

    def synchronize(self):
        """
        Synchronize the current device
        """
        libcuda.synchronizeDevice()
        return

    # debugging
    def dump(self, indent=''):
        """
        Print information about this device
        """
        print(f"{indent}device {self.id}:")
        print(f"{indent}  name: {self.name}")

        print(f"{indent}  driver version: {self.driverVersion}")
        print(f"{indent}  runtime version: {self.runtimeVersion}")
        print(f"{indent}  compute capability: {self.capability}")
        print(f"{indent}  compute mode: {self.computeMode}")

        print(f"{indent}  managed memory: {self.managedMemory}")
        print(f"{indent}  unified addressing: {self.unifiedAddressing}")

        print(f"{indent}  processors: {self.processors}")
        print(f"{indent}  cores per processor: {self.coresPerProcessor}")

        print(f"{indent}  global memory: {self.globalMemory} bytes")
        print(f"{indent}  constant memory: {self.constantMemory} bytes")
        print(f"{indent}  shared memory per block: {self.sharedMemoryPerBlock} bytes")

        print(f"{indent}  warp: {self.warp} threads")
        print(f"{indent}  max threads per block: {self.maxThreadsPerBlock}")
        print(f"{indent}  max threads per processor: {self.maxThreadsPerProcessor}")

        print(f"{indent}  max dimensions of a grid: {self.maxGrid}")
        print(f"{indent}  max dimensions of a thread block: {self.maxThreadBlock}")

        # all done
        return


# end of file
