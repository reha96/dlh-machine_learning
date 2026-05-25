#!/usr/bin/env python3
"""Write a Python function that inserts a new document in a collection
based on kwargs:
    """


def insert_school(mongo_collection, **kwargs):
    """function that inserts a new document

    Args:
        mongo_collection (_type_): _description_

    Returns:
        _type_: _description_
    """
    return mongo_collection.insert(kwargs)
