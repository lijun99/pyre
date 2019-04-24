# -*- coding: utf-8 -*-
#
# Lijun Zhu
# california institute of technology
# (c) 2016-2019  all rights reserved
#

####### Not working, python thread is not holding a cuda hook

# externals
from . import cuda as libcuda # the extension

class Timer:
    """
    A cuda timer using cudaEvent
    """
    # a python capsule for c++ timer object
    capsule = None

    def __init__(self, **kwds):
        """
        create a cuda timer
        """
        self.capsule = libcuda.timer_alloc()
        return

    def start(self):
        libcuda.timer_start(self.capsule)
        return

    def stop(self):
        libcuda.timer_start(self.capsule)
        return

    def time(self, process=None):
        elapsedtime = libcuda.timer_time(self.capsule)
        print(f'The cuda process {process} takes {elapsedtime} ms.')
        return

#end of file
