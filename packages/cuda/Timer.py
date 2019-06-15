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
        libcuda.timer_stop(self.capsule)
        elapsedtime = libcuda.timer_time(self.capsule)
        return elapsedtime

    def profile(self, process, *args, **kwargs):
        """
        Profile a process
        :param process:
        :param args:
        :return:
        """
        # start the timer
        self.start()
        process(*args, **kwargs)
        elapsedtime = self.stop()
        return elapsedtime

#end of file
