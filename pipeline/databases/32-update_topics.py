#!/usr/bin/env python3
"""Write a Python function that changes all topics
    of a school document based on the name:
    """


def update_topics(mongo_collection, name, topics):
    """changes all topics
    of a school document

    Args:
        mongo_collection (_type_): _description_
        name (_type_): _description_
        topics (_type_): _description_

    Returns:
        _type_: _description_
    """
    return mongo_collection.update(
        {'name': name}, {'$set': {'topics': topics}}, {'multi': True}
    )
