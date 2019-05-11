# -*- coding: utf-8 -*-
#
# Lijun Zhu
# california institute of technology
# (c) 2016-2019  all rights reserved
#

# externals
from . import cuda as libcuda # the extension

from .Matrix import Matrix

class cuSolverDn:
    """
    Wrapper for cusolverDn lib utitilies
    """

    def create_handle():
        """
        create a cusolverDn handle
        """
        handle = libcuda.cusolverDnCreate()
        return handle

    def get_current_handle():
        # default device handle
        from . import manager
        if manager.current_device is None:
            manager.device(0)
        handle = manager.current_device.cusolverdn_handle
        return handle

# end of file
