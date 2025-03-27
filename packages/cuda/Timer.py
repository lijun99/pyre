# -*- coding: utf-8 -*-
#
# Lijun Zhu
# california institute of technology
# (c) 2016-2019  all rights reserved
#

####### Not working, python thread is not holding a cuda hook

# externals
from . import cuda as libcuda # the extension
import contextlib

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

    @contextlib.contextmanager
    def profile(self):
        """
        Profile a block of code using a context manager.

        Example usage:
        with timer.profile():
            # Code to be timed
            my_function(arg1, arg2)
            another_operation()
        elapsed_time = timer.elapsed_time
        """
        self.start()
        yield
        self.elapsed_time = self.stop()

#end of file
