# -*- coding: utf-8 -*-
#
# Lijun Zhu
# california institute of technology
# (c) 2016-2019  all rights reserved
#


# externals
from . import cuda as libcuda # the extension
import pyre
from .Matrix import Matrix
from .Vector import Vector


class cuRand:
    """
    CURAND lib utitilies
    """

    # curand generator types
    CURAND_RNG_TEST = 0,
    CURAND_RNG_PSEUDO_DEFAULT = 100 # Default pseudorandom generator
    CURAND_RNG_PSEUDO_XORWOW = 101 # XORWOW pseudorandom generator
    CURAND_RNG_PSEUDO_MRG32K3A = 121 # MRG32k3a pseudorandom generator
    CURAND_RNG_PSEUDO_MTGP32 = 141 # Mersenne Twister MTGP32 pseudorandom generator
    CURAND_RNG_PSEUDO_MT19937 = 142 # Mersenne Twister MT19937 pseudorandom generator
    CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161 # PHILOX-4x32-10 pseudorandom generator
    CURAND_RNG_QUASI_DEFAULT = 200#  Default quasirandom generator
    CURAND_RNG_QUASI_SOBOL32 = 201#  Sobol32 quasirandom generator
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202 # Scrambled Sobol32 quasirandom generator
    CURAND_RNG_QUASI_SOBOL64 = 203 #  Sobol64 quasirandom generator
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204 # Scrambled Sobol64 quasirandom generator

    def create_generator(gentype=None, seed=None):
        """
        allocate a curand generator 
        """
        # set the generator type
        if gentype is None:
            gentype = cuRand.CURAND_RNG_PSEUDO_DEFAULT
        # create the generator gen    
        gen =  libcuda.curand_alloc(gentype)
        # set seed if available
        if seed is not None:
            libcuda.curand_setseed(gen, seed)
        # all done
        return gen

    def set_seed(gen, seed):
        """
        Set seed for curand generator
        """
        libcuda.curand_setseed(gen, seed)
        return self
    
    def get_current_generator():
        """
        Find the curand generator from current device
        """
        from . import manager 
        gen = manager.current_device.get_curand_generator()
        return gen
    

    def gaussian(gen=None, out=None, dtype='float64', loc=0, scale=1, size=1):
        """
        generate Gaussian(Normal) distribution random numbers 
        """
        if gen is None:
            gen = cuRand.get_current_generator()
    
        if out is None:
            if isinstance(size, int):
                out = Vector(shape=size, dtype=dtype)
            elif isinstance(size, tuple) and len(size) == 2:
                out = Matrix(shape=size, dtype=dtype)
            else:
                raise NotImplementedError(f'size {size} is not an int or 2-tuple')
                return 
        # call curand        
        libcuda.curand_gaussian(gen, out.data, loc, scale)           
        # return
        return out

    def uniform(gen=None, out=None, dtype='float64', size=1):
        """
        generate uniform distribution random numbers (0,1] 
        """
        if gen is None:
            gen = cuRand.get_current_generator()
        
        if out is None:
            if isinstance(size, int):
                out = Vector(shape=size, dtype=dtype)
            elif isinstance(size, tuple) and len(size) == 2:
                out = Matrix(shape=size, dtype=dtype)

            else:
                raise NotImplementedError(f'size {size} is not an int or 2-tuple')
                return 
        # call curand        
        libcuda.curand_uniform(gen, out.data)           
        # return
        return out

# end of file
