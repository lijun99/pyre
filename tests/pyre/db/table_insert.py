#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2012 all rights reserved
#


"""
Exercise inserting rows in tables
"""


def test():
    # access the package
    import pyre.db

    # declare the person table
    class Person(pyre.db.table, id='persons'):

        id = pyre.db.int().primary()
        name = pyre.db.str().notNull()
        phone = pyre.db.str(maxlen=10).notNull()

    # declare the customer table
    class Customer(pyre.db.table, id='customers'):
        """
        Simple customer table
        """
        # the data fields
        cid = pyre.db.int().primary()
        pid = pyre.db.reference(key=Person.id)
        balance = pyre.db.decimal(precision=7, scale=2).setDefault(0)


    # create some customers
    customers = [
        Person(id=107, name="Bit Twiddle", phone="+1 800 555 1114"),
        Person(id=108, name="Eva Lu Ator", phone="+1 800 555 7687"),
        Customer(cid=1023, pid=107, balance=1000),
        Customer(cid=1024, pid=108, balance=50),
        ]

    # get a server
    server = pyre.db.server(name="test")

    # generate the SQL statement that creates the customer table
    stmt = tuple(server.sql.insertRecords(*customers))
    # print('\n'.join(stmt))
    assert stmt == (
        "INSERT INTO persons",
        "    (id, name, phone)",
        "  VALUES",
        "    (107, 'Bit Twiddle', '+1 800 555 1114'),",
        "    (108, 'Eva Lu Ator', '+1 800 555 7687');",
        "INSERT INTO customers",
        "    (cid, pid, balance)",
        "  VALUES",
        "    (1023, 107, 1000),",
        "    (1024, 108, 50);",
        )

    return


# main
if __name__ == "__main__":
    test()


# end of file 
