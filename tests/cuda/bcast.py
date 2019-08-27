# -*- coding: utf-8 -*-
#
# Lijun Zhu (ljzhu@gps.caltech.edu)
#
# (c) 2018-2019 all rights reserved
#


"""
Exercise the mpi_bcast of cuda vector and matrix
"""


def test():
    # setup the workload
    samples = 8
    value = 7.0

    # externals
    import mpi
    import cuda

    mpi.init()
    # get the world communicator
    world = mpi.world
    # figure out its geometry
    rank = world.rank
    tasks = world.size

    # decide which task is the source
    source = 0

    # vector test
    v = cuda.vector(shape=samples)

    # set values at the source task
    if rank == source:
        v.fill(value)

    # broadcast
    v.bcast(communicator=world, source=source)

    # verify that i got the correct part
    cv = v.copy_to_host()
    for index in range(samples):
        assert cv[index] == value


    # matrix test
    m = cuda.matrix(shape=(samples, samples))

    # set values at the source task
    if rank == source:
        m.fill(value)

    # broadcast
    m.bcast(communicator=world, source=source)

    # verify that i got the correct part
    cm = m.copy_to_host()
    for i in range(samples):
        for j in range(samples):
            assert cm[i,j] == value

    # all done
    return


# main
if __name__ == "__main__":
    # do...
    test()


# end of file
