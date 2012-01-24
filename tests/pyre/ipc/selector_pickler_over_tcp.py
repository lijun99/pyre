#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2012 all rights reserved
#


"""
Exercise a selector watching over file descriptors
"""

# externals
import os
import pyre.ipc

# if necessary
import journal
serverdbg = journal.debug("selector.server")
# serverdbg.active = True
clientdbg = journal.debug("selector.client")
# clientdbg.active = True


def test():
    # build the marshaller
    m = pyre.ipc.pickler()
    # and the communication channels
    server, client = pyre.ipc.pipe()
    
    # fork
    pid = os.fork()
    # in the server process
    if pid > 0:
        # invoke the server behavior
        return onServer(clientPid=pid, marshaller=m, pipe=client)

    # in the client process
    # get my pid
    clientPid = os.getpid()
    # invoke the behavior
    return onClient(clientPid=clientPid, marshaller=m, pipe=server)


def onServer(clientPid, marshaller, pipe):
    # observe the server selector at work
    # journal.debug("pyre.ipc.selector").active = True

    # establish a network presence
    port = pyre.ipc.port(address='localhost:0')
    # report what it was bound to
    serverdbg.log("server: host={!r}, port={}".format(*port.address.value))

    def getMessage(selector, channel, **kwds):
        message = marshaller.recv(channel)
        serverdbg.log("server: received {!r}".format(message)) 
        # check it
        assert message == "Hello from {}".format(clientPid)
        return False

    def sendPortNumber(selector, channel, **kwds):
        serverdbg.log("server: sending port {!r}".format(port))
        marshaller.send(channel=channel, item=port.address.port)
        serverdbg.log("server: done sending port")
        return False

    def connectionAttempt(selector, channel, **kwds):
        peer, address = port.accept()
        serverdbg.log("server: connection attempt from {!r}:{}".format(*address.value))
        # schedule the receiving of the message
        selector.notifyOnReadReady(channel=peer, handler=getMessage)
        # and stop waiting for any further connections
        return False

    # build my selector
    serverdbg.log("server: building a selector")
    s = pyre.ipc.selector()
    # let me know when the pipe to the client is ready for writing so i can send my port
    serverdbg.log("server: registering the port notification routine")
    s.notifyOnWriteReady(channel=pipe, handler=sendPortNumber)
    serverdbg.log("server: registering the connection routine")
    s.notifyOnReadReady(channel=port.channel, handler=connectionAttempt)

    # invoke the selector
    serverdbg.log("server: entering watch")
    s.watch()
    serverdbg.log("server: all done")

    # all done
    return


def onClient(clientPid, marshaller, pipe):
    # observe the client selector at work
    # journal.debug("pyre.ipc.selector").active = True

    # the port notification routine
    def recvPortNumber(selector, channel, **kwds):
        # get the port
        clientdbg.log("client: receiving port number")
        port = marshaller.recv(channel)
        clientdbg.log("client: port={!r}".format(port))
        
        # make a connection
        tcp = pyre.ipc.tcp(address="localhost:{}".format(port))
        # send a message
        message = "Hello from {}".format(clientPid)
        clientdbg.log("client: sending {!r}".format(message))
        marshaller.send(channel=tcp, item=message)
        # all done
        return False

    # build my selector
    clientdbg.log("client: building a selector")
    s = pyre.ipc.selector()
    # let me know when the pipe to the client is ready for writing so i can send my port
    clientdbg.log("client: registering the port notification routine")
    s.notifyOnReadReady(channel=pipe, handler=recvPortNumber)

    # invoke the selector
    clientdbg.log("client: entering watch")
    s.watch()
    clientdbg.log("client: all done")

    # all done
    return
    

# main
if __name__ == "__main__":
    test()


# end of file 