# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# meta-class
from pyre.patterns.Singleton import Singleton
# the extension with CUDA support
from . import cuda as libcuda


# declaration
class DeviceManager(metaclass=Singleton):
    """
    The singleton that provides access to what is known about CUDA capable hardware
    """


    # public data
    count = 0
    devices = []
    
    # manager keeps track of the current device
    # current_device can be switched by cuda.device(0), cuda.device(1) ...
    current_device = None


    # interface
    def device(self, did=0):
        """
        Set {did} as the default device
        """
        self.current_device=self.devices[did]
        self.current_device.initialize()
        # delegate to the extension module
        
        return self.current_device


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)

        # grab the device class
        from .Device import Device
        # build the device list and attach it
        self.devices = libcuda.discover(Device)
        # set the count
        self.count = len(self.devices)
        # set a default device
        if self.count > 0: 
            self.current_device = self.device() 
        # all done
        return


# end of file
